"""DNABERT-2: Efficient Foundation Model for Multi-Species Genome.

This module implements a DeepChem wrapper for DNABERT-2, a multi-species
DNA language model based on BERT with BPE tokenization and ALiBi positional
embeddings.

Reference
---------
Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R., & Liu, H. (2023).
DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.
arXiv:2306.15006. https://arxiv.org/abs/2306.15006
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from deepchem.models.torch_models.hf_models import HuggingFaceModel

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False

logger = logging.getLogger(__name__)

# Default pretrained model on HuggingFace Hub
_DNABERT2_DEFAULT_MODEL_PATH = "zhihan1996/DNABERT-2-117M"


class DNABERT2(HuggingFaceModel):
    """DNABERT-2 model for DNA sequence analysis.

    DNABERT-2 is a foundation model pretrained on large-scale multi-species
    genomes. It improves upon the original DNABERT [1] by replacing k-mer
    tokenization with Byte Pair Encoding (BPE) and standard learnable positional
    embeddings with Attention with Linear Biases (ALiBi), resulting in better
    efficiency and multi-species genome understanding.

    This class wraps the pretrained ``zhihan1996/DNABERT-2-117M`` (or any
    compatible model checkpoint) from the HuggingFace Hub as a DeepChem model,
    supporting the full ``fit`` / ``predict`` / ``evaluate`` API.

    The model supports the following tasks:

    - ``mlm`` — Masked Language Modeling for pretraining / continued pretraining.
    - ``regression`` — Single or multi-output regression (e.g., predicting
      chromatin accessibility scores).
    - ``classification`` — Single-label or multi-label classification (e.g.,
      promoter / enhancer / splice-site classification).
    - ``mtr`` — Multi-task regression (simultaneous prediction of multiple
      continuous targets).
    - ``feature_extractor`` — Return backbone CLS-token embeddings without a
      task head; use ``get_embeddings()`` to extract them.

    **Data format**: raw DNA sequences (strings of A/C/G/T) are stored in
    the ``X`` field of a ``dc.data.NumpyDataset`` or ``DiskDataset``. Use
    ``dc.feat.DummyFeaturizer()`` when loading from CSV so that sequences are
    passed through unmodified.

    Parameters
    ----------
    task : str
        Learning task. One of ``'mlm'``, ``'regression'``, ``'classification'``,
        ``'mtr'``, or ``'feature_extractor'``.
    model_path : str, optional (default ``'zhihan1996/DNABERT-2-117M'``)
        HuggingFace Hub model ID or path to a local directory containing a
        saved DNABERT-2 checkpoint.  Must be usable with
        ``AutoModel.from_pretrained(..., trust_remote_code=True)``.
    n_tasks : int, optional (default 1)
        Number of output tasks/labels. Used to set ``num_labels`` in the
        sequence classification head.
    config : dict, optional (default ``{}``)
        Additional keyword arguments forwarded to ``AutoConfig`` when building
        the model from a local/Hub checkpoint via ``load_from_pretrained``.
    max_seq_length : int, optional (default 512)
        Maximum tokenized sequence length. Sequences longer than this value
        are silently truncated.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import shutil
    >>> import numpy as np
    >>> import deepchem as dc

    >>> tempdir = tempfile.mkdtemp()

    >>> # --- Build a tiny DNA dataset ---
    >>> sequences = np.array([
    ...     "ATCGATCGATCGATCG",
    ...     "GCTAGCTAGCTAGCTA",
    ...     "AAAAGGGGCCCCTTTT",
    ...     "TTTTCCCCGGGGAAAA",
    ... ])
    >>> labels = np.array([[0], [1], [0], [1]])
    >>> dataset = dc.data.NumpyDataset(X=sequences, y=labels)

    >>> # --- MLM pretraining ---
    >>> from deepchem.models.torch_models.dna_bert2 import DNABERT2
    >>> pretrain_dir = os.path.join(tempdir, 'pretrain')
    >>> pretrain_model = DNABERT2(task='mlm', model_dir=pretrain_dir)
    >>> loss = pretrain_model.fit(dataset, nb_epoch=1)
    >>> assert loss > 0.0

    >>> # --- Transfer to classification ---
    >>> finetune_dir = os.path.join(tempdir, 'finetune')
    >>> finetune_model = DNABERT2(task='classification', n_tasks=1,
    ...                           model_dir=finetune_dir)
    >>> _ = finetune_model.load_from_pretrained(pretrain_dir)
    >>> loss = finetune_model.fit(dataset, nb_epoch=1)
    >>> preds = finetune_model.predict(dataset)
    >>> assert preds.shape == (len(sequences), 2)

    >>> # --- Cleanup ---
    >>> if os.path.exists(tempdir):
    ...     shutil.rmtree(tempdir)

    References
    ----------
    .. [1] Zhou, Z. et al. DNABERT-2: Efficient Foundation Model and
       Benchmark for Multi-Species Genome. arXiv:2306.15006 (2023).
    .. [2] Ji, Y. et al. DNABERT: pre-trained Bidirectional Encoder
       Representations from Transformers model for DNA-language in genome.
       Bioinformatics 37(15), 2112-2120 (2021).
    """

    def __init__(
            self,
            task: str,
            model_path: str = _DNABERT2_DEFAULT_MODEL_PATH,
            n_tasks: int = 1,
            config: Optional[Dict[str, Any]] = None,
            max_seq_length: int = 512,
            **kwargs) -> None:
        """Initialise DNABERT2 wrapper.

        Parameters
        ----------
        task : str
            See class docstring.
        model_path : str
            HuggingFace Hub model ID or local checkpoint path.
        n_tasks : int
            Number of output labels / regression targets.
        config : dict, optional
            Extra configuration overrides forwarded to ``AutoConfig``.
        max_seq_length : int
            Upper bound on tokenised sequence length (sequences are truncated).
        **kwargs
            Forwarded verbatim to :class:`~deepchem.models.torch_models.hf_models.HuggingFaceModel`.
        """
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        from transformers.modeling_utils import PreTrainedModel

        self.n_tasks = n_tasks
        self.max_seq_length = max_seq_length
        config_dict: Dict[str, Any] = config if config is not None else {}

        # ------------------------------------------------------------------
        # Tokenizer
        # ------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            model_max_length=max_seq_length,
        )

        # ------------------------------------------------------------------
        # Task-conditioned model head
        # ------------------------------------------------------------------
        model: PreTrainedModel

        if task == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **config_dict,
            )

        elif task in ("regression", "mtr"):
            hf_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                num_labels=n_tasks,
                problem_type="regression",
                **config_dict,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=hf_config,
                trust_remote_code=True,
            )

        elif task == "classification":
            if n_tasks == 1:
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
            hf_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                num_labels=n_tasks if n_tasks > 1 else 2,
                problem_type=problem_type,
                **config_dict,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=hf_config,
                trust_remote_code=True,
            )

        elif task == "feature_extractor":
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                **config_dict,
            )

        else:
            raise ValueError(
                f"Invalid task '{task}'. Supported tasks: 'mlm', 'regression', "
                f"'mtr', 'classification', 'feature_extractor'.")

        super(DNABERT2, self).__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(
            self,
            batch: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
        """Prepare a batch of DNA sequences for forward pass.

        Overrides the parent :meth:`HuggingFaceModel._prepare_batch` to add
        DNA-specific preprocessing:

        * Sequences are uppercased and have ambiguous bases (``N``) removed.
        * Tokenization uses ``truncation=True`` with ``max_length`` equal to
          ``self.max_seq_length`` to guard against excessively long sequences.
        * Label dtype follows the same convention as ``Chemberta``:
          ``float`` for regression / mtr, ``long`` for single-label
          classification, ``float`` for multi-label classification.

        Parameters
        ----------
        batch : tuple
            A ``(inputs, labels, weights)`` tuple generated by the dataset
            iterator.  ``inputs[0]`` is a numpy array of raw DNA strings.

        Returns
        -------
        tuple
            ``(tokenized_inputs_dict, labels_tensor_or_None, weights)``
        """
        from transformers.data.data_collator import DataCollatorForLanguageModeling

        sequences_batch, y, w = batch
        # DNA-specific preprocessing: uppercase + strip ambiguous 'N' bases
        sequences: List[str] = [
            seq.upper().replace("N", "")
            for seq in sequences_batch[0].tolist()
        ]

        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        if self.task == "mlm":
            # data_collator is set by HuggingFaceModel.__init__ when task='mlm'
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens["input_ids"])
            inputs_dict = {
                "input_ids": inputs.to(self.device),
                "labels": labels.to(self.device),
                "attention_mask": tokens["attention_mask"].to(self.device),
            }
            return inputs_dict, None, w

        elif self.task in ("regression", "classification", "mtr"):
            if y is not None:
                y_tensor = torch.from_numpy(y[0])
                if self.task in ("regression", "mtr"):
                    y_tensor = y_tensor.float().to(self.device)
                elif self.task == "classification":
                    if self.n_tasks == 1:
                        # CrossEntropyLoss expects long (class indices)
                        y_tensor = y_tensor.long().to(self.device)
                    else:
                        # BCEWithLogitsLoss expects float
                        y_tensor = y_tensor.float().to(self.device)
            else:
                y_tensor = None  # type: ignore[assignment]

            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs_dict = {**tokens, "labels": y_tensor}
            return inputs_dict, y_tensor, w

        # feature_extractor: batch preparation mirrors regression but without labels
        else:
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)
            return tokens, None, w

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def get_embeddings(
            self,
            input_ids: "torch.Tensor",
            attention_mask: "torch.Tensor") -> "torch.Tensor":
        """Extract CLS-token embeddings from the DNABERT-2 backbone.

        This method is intended for use with ``task='feature_extractor'``.
        It performs a forward pass through the backbone (no task head) and
        returns the representation at position 0 (the ``[CLS]`` token) of
        the last transformer layer.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor
            Attention mask of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            CLS embeddings of shape ``(batch, hidden_size)``.

        Raises
        ------
        RuntimeError
            If the model task is not ``'feature_extractor'``.

        Examples
        --------
        >>> import torch
        >>> from deepchem.models.torch_models.dna_bert2 import DNABERT2
        >>> model = DNABERT2(task='feature_extractor')  # doctest: +SKIP
        >>> tok = model.tokenizer("ATCGATCG", return_tensors='pt')  # doctest: +SKIP
        >>> emb = model.get_embeddings(tok['input_ids'], tok['attention_mask'])  # doctest: +SKIP
        >>> emb.shape  # (1, 768)  # doctest: +SKIP
        """
        if self.task != "feature_extractor":
            raise RuntimeError(
                "get_embeddings() is only available when task='feature_extractor'. "
                f"Current task: '{self.task}'.")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Return the CLS token (position 0) of the last hidden layer
        return outputs.last_hidden_state[:, 0, :]
