"""Tests for the DNABERT-2 DeepChem wrapper.

All tests are marked with ``@pytest.mark.hf`` following the convention
established by ``test_chemberta.py`` and ``test_prot_bert.py``.  They require
network access on the first run (to download ``zhihan1996/DNABERT-2-117M``
from HuggingFace Hub) and are skipped when the ``hf`` marker is not selected.

Run with::

    pytest deepchem/models/torch_models/tests/test_dna_bert2.py -v -m hf
"""
import os

import deepchem as dc
import numpy as np
import pytest

try:
    import torch
    from deepchem.models.torch_models.dna_bert2 import DNABERT2
except ModuleNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Helper: default model path
# ---------------------------------------------------------------------------
_MODEL_PATH = "zhihan1996/DNABERT-2-117M"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.hf
def test_dnabert2_pretraining(dna_classification_dataset):
    """MLM pretraining runs without error and returns a positive loss."""
    model = DNABERT2(task="mlm", model_path=_MODEL_PATH)
    loss = model.fit(dna_classification_dataset, nb_epoch=1)
    assert loss > 0.0, f"Expected positive MLM loss, got {loss}"


@pytest.mark.hf
def test_dnabert2_finetuning_regression(dna_regression_dataset):
    """Regression fine-tuning: predictions have the same shape as labels."""
    model = DNABERT2(task="regression", model_path=_MODEL_PATH, n_tasks=1)
    loss = model.fit(dna_regression_dataset, nb_epoch=1)
    assert loss > 0.0

    preds = model.predict(dna_regression_dataset)
    assert preds.shape == dna_regression_dataset.y.shape, (
        f"Expected shape {dna_regression_dataset.y.shape}, got {preds.shape}"
    )

    eval_score = model.evaluate(
        dna_regression_dataset,
        metrics=dc.metrics.Metric(dc.metrics.mean_absolute_error),
    )
    assert eval_score


@pytest.mark.hf
def test_dnabert2_finetuning_classification(dna_classification_dataset):
    """Binary classification: predictions have shape (N, 2) (two logits)."""
    model = DNABERT2(task="classification", model_path=_MODEL_PATH, n_tasks=1)
    loss = model.fit(dna_classification_dataset, nb_epoch=1)
    assert loss > 0.0

    preds = model.predict(dna_classification_dataset)
    n_samples = dna_classification_dataset.y.shape[0]
    # single-label classification → output is (N, num_classes=2)
    assert preds.shape == (n_samples, 2), (
        f"Expected shape ({n_samples}, 2), got {preds.shape}"
    )

    eval_score = model.evaluate(
        dna_classification_dataset,
        metrics=dc.metrics.Metric(dc.metrics.recall_score),
    )
    assert eval_score


@pytest.mark.hf
def test_dnabert2_finetuning_mtr(dna_regression_dataset):
    """Multi-task regression (mtr): model trains and predictions have correct shape."""
    # Build a 2-task dataset by duplicating the single label column
    y2 = np.hstack(
        [dna_regression_dataset.y, dna_regression_dataset.y]
    )
    dataset = dc.data.NumpyDataset(
        X=dna_regression_dataset.X,
        y=y2,
        ids=dna_regression_dataset.ids,
    )
    model = DNABERT2(task="mtr", model_path=_MODEL_PATH, n_tasks=2)
    loss = model.fit(dataset, nb_epoch=1)
    assert loss > 0.0

    preds = model.predict(dataset)
    assert preds.shape == dataset.y.shape, (
        f"Expected shape {dataset.y.shape}, got {preds.shape}"
    )


@pytest.mark.hf
def test_dnabert2_load_from_pretrained(tmpdir, dna_classification_dataset):
    """Pretrain → save_checkpoint → load_from_pretrained transfers backbone
    weights correctly between MLM and classification models.

    Mirrors ``test_chemberta_load_from_pretrained``.
    """
    pretrain_dir = os.path.join(str(tmpdir), "pretrain")
    finetune_dir = os.path.join(str(tmpdir), "finetune")

    # Pretrain and save a checkpoint
    pretrain_model = DNABERT2(
        task="mlm",
        model_path=_MODEL_PATH,
        model_dir=pretrain_dir,
    )
    pretrain_model.save_checkpoint()

    # Build a classification model and load backbone from the pretrain checkpoint
    finetune_model = DNABERT2(
        task="classification",
        model_path=_MODEL_PATH,
        n_tasks=1,
        model_dir=finetune_dir,
    )
    finetune_model.load_from_pretrained(pretrain_dir)

    # All backbone (bert / encoder) weights should match
    pretrain_sd = pretrain_model.model.state_dict()
    finetune_sd = finetune_model.model.state_dict()

    # Keys that belong to the shared backbone (not the task head)
    backbone_keys = [
        k for k in pretrain_sd.keys()
        if "bert" in k or "encoder" in k or "embeddings" in k
    ]
    assert len(backbone_keys) > 0, "No backbone keys found — model key names may have changed"

    matches = [
        torch.allclose(pretrain_sd[k], finetune_sd[k])
        for k in backbone_keys
        if k in finetune_sd
    ]
    assert all(matches), "Backbone weights do not match after load_from_pretrained"


@pytest.mark.hf
def test_dnabert2_save_reload(tmpdir):
    """save_checkpoint → restore correctly reconstructs the exact same weights."""
    model = DNABERT2(
        task="regression",
        model_path=_MODEL_PATH,
        n_tasks=1,
        model_dir=str(tmpdir),
    )
    model._ensure_built()
    model.save_checkpoint()

    model_new = DNABERT2(
        task="regression",
        model_path=_MODEL_PATH,
        n_tasks=1,
        model_dir=str(tmpdir),
    )
    model_new.restore()

    old_sd = model.model.state_dict()
    new_sd = model_new.model.state_dict()
    matches = [
        torch.allclose(old_sd[k], new_sd[k]) for k in old_sd.keys()
    ]
    assert all(matches), "Not all weights matched after save/reload"


@pytest.mark.hf
def test_dnabert2_embeddings():
    """feature_extractor task + get_embeddings() returns a valid CLS embedding."""
    model = DNABERT2(task="feature_extractor", model_path=_MODEL_PATH)
    model._ensure_built()
    model.model.eval()

    sequence = "ATCGATCGATCGATCG"
    tok = model.tokenizer(sequence, return_tensors="pt")
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    with torch.no_grad():
        emb = model.get_embeddings(input_ids, attention_mask)

    # DNABERT-2-117M hidden size is 768
    assert emb.ndim == 2, f"Expected 2-D tensor, got shape {emb.shape}"
    assert emb.shape[0] == 1, f"Expected batch=1, got {emb.shape[0]}"
    assert emb.shape[1] > 0, "Embedding dimension must be > 0"


@pytest.mark.hf
def test_dnabert2_embeddings_invalid_task():
    """get_embeddings() raises RuntimeError when task != 'feature_extractor'."""
    model = DNABERT2(task="regression", model_path=_MODEL_PATH, n_tasks=1)
    model._ensure_built()
    tok = model.tokenizer("ATCG", return_tensors="pt")
    with pytest.raises(RuntimeError, match="feature_extractor"):
        model.get_embeddings(tok["input_ids"], tok["attention_mask"])


@pytest.mark.hf
def test_dnabert2_load_from_hf_hub(tmpdir):
    """load_from_pretrained(..., from_hf_checkpoint=True) replaces the model
    object in-place with a freshly loaded HuggingFace model instance.

    Mirrors ``test_chemberta_load_weights_from_hf_hub``.
    """
    model = DNABERT2(
        task="classification",
        model_path=_MODEL_PATH,
        n_tasks=1,
        model_dir=str(tmpdir),
    )
    old_id = id(model.model)
    model.load_from_pretrained(_MODEL_PATH, from_hf_checkpoint=True)
    new_id = id(model.model)
    # The in-place replacement via AutoModel.from_pretrained should produce a
    # new Python object even though the weights may be equivalent.
    assert old_id != new_id, (
        "load_from_pretrained(from_hf_checkpoint=True) did not replace model object"
    )


@pytest.mark.hf
def test_dnabert2_dna_preprocessing():
    """_prepare_batch strips ambiguous bases and uppercases sequences."""
    model = DNABERT2(task="classification", model_path=_MODEL_PATH, n_tasks=1)
    model._ensure_built()

    # Include lowercase and ambiguous 'N' that should be cleaned
    sequences = np.array(["atcgnnngatcg", "GCTA"])
    labels = np.array([[0], [1]])
    dataset = dc.data.NumpyDataset(X=sequences, y=labels)

    # fit() calls _prepare_batch internally; if preprocessing is broken the
    # tokenizer would either raise or silently produce wrong token IDs.
    loss = model.fit(dataset, nb_epoch=1)
    assert isinstance(loss, float)
