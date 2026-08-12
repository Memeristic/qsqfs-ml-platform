"""Multimodal encoders, fusion, and the manifest data path.

These are the tests that let the encoders exist at all: the README's rule is
that untested code implying an untested capability does not ship.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.multimodal import build_schema, collate, infer_modalities
from src.models.encoders import GenomicEncoder, ImageEncoder, TabularEncoder, TextEncoder
from src.models.fusion import MultimodalFusion
from src.models.multimodal import MultimodalModel

EMB = 32


# ------------------------------------------------------------- encoders
def test_tabular_encoder_shape():
    assert TabularEncoder(20, EMB)(torch.randn(4, 20)).shape == (4, EMB)


@pytest.mark.parametrize("shape", [(2, 3, 32, 32), (2, 1, 32, 32), (2, 4, 3, 32, 32)])
def test_image_encoder_accepts_rgb_grayscale_and_volumes(shape):
    encoder = ImageEncoder(backbone="small_cnn", embedding_dim=EMB)
    assert encoder(torch.randn(*shape)).shape == (2, EMB)


def test_image_encoder_reports_whether_it_is_actually_pretrained():
    """describe() must never imply an ImageNet prior that was not loaded."""
    encoder = ImageEncoder(backbone="small_cnn", embedding_dim=EMB)
    info = encoder.describe()
    assert info["pretrained"] is False
    assert info["backbone"] == "small_cnn"


def test_image_encoder_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="Unsupported backbone"):
        ImageEncoder(backbone="not_a_real_model")


@pytest.mark.parametrize("kind,x,dim", [
    ("expression", torch.randn(4, 30), 30),
    ("snp", torch.randint(0, 3, (4, 30)), 30),
    ("sequence", torch.randint(1, 5, (4, 60)), 60),
])
def test_genomic_encoder_all_three_encodings(kind, x, dim):
    encoder = GenomicEncoder(dim, kind, embedding_dim=EMB, vocab_size=5)
    assert encoder(x).shape == (4, EMB)


def test_genomic_encoder_requires_vocab_for_sequences():
    with pytest.raises(ValueError, match="vocab_size"):
        GenomicEncoder(60, "sequence", embedding_dim=EMB)


def test_text_encoder_fallback_is_deterministic_and_declares_itself():
    encoder = TextEncoder(embedding_dim=EMB, force_fallback=True)
    info = encoder.describe()
    assert info["pretrained"] is False
    assert "no pretrained clinical knowledge" in info["caveat"]
    encoder.eval()
    texts = ["chest pain on exertion", "routine review"]
    with torch.no_grad():
        assert torch.allclose(encoder(texts), encoder(texts))


def test_hashed_bow_is_stable_across_calls():
    from src.models.encoders.text import hashed_bag_of_words

    a = hashed_bag_of_words(["patient stable"], 256)
    b = hashed_bag_of_words(["patient stable"], 256)
    assert torch.equal(a, b)
    assert a.sum() > 0


# --------------------------------------------------------------- fusion
@pytest.mark.parametrize("fusion_type", ["early", "late", "hybrid"])
def test_fusion_types_produce_correct_output_shape(fusion_type):
    fusion = MultimodalFusion({"a": EMB, "b": EMB}, fusion_type, hidden_dim=EMB, n_outputs=2)
    out = fusion({"a": torch.randn(5, EMB), "b": torch.randn(5, EMB)})
    assert out.shape == (5, 2)


def test_late_fusion_exposes_readable_modality_weights():
    fusion = MultimodalFusion({"a": EMB, "b": EMB}, "late", hidden_dim=EMB)
    weights = fusion.modality_weights()
    assert set(weights) == {"a", "b"}
    assert pytest.approx(sum(weights.values()), abs=1e-5) == 1.0
    assert MultimodalFusion({"a": EMB}, "early", hidden_dim=EMB).modality_weights() is None


def test_missing_modality_differs_from_a_zero_measurement():
    """A row with no image must not be treated as an all-zero image."""
    fusion = MultimodalFusion({"a": EMB, "b": EMB}, "early", hidden_dim=EMB)
    fusion.eval()
    embeddings = {"a": torch.randn(3, EMB), "b": torch.randn(3, EMB)}
    with torch.no_grad():
        present = fusion(embeddings, {"b": torch.tensor([1.0, 1.0, 1.0])})
        absent = fusion(embeddings, {"b": torch.tensor([1.0, 1.0, 0.0])})
    assert torch.allclose(present[:2], absent[:2], atol=1e-5)
    assert not torch.allclose(present[2], absent[2], atol=1e-5)


def test_fusion_rejects_bad_head_divisor_and_missing_tensors():
    with pytest.raises(ValueError):
        MultimodalFusion({"a": 30}, "hybrid", hidden_dim=30, n_heads=4)
    fusion = MultimodalFusion({"a": EMB, "b": EMB}, "early", hidden_dim=EMB)
    with pytest.raises(KeyError):
        fusion({"a": torch.randn(2, EMB)})


# ----------------------------------------------------- end-to-end model
def test_multimodal_model_forward_and_description():
    schema = {
        "labs": {"type": "tabular", "input_dim": 12},
        "notes": {"type": "text", "force_fallback": True},
        "scan": {"type": "image", "backbone": "small_cnn"},
        "expr": {"type": "genomic", "input_dim": 20, "encoding": "expression"},
    }
    model = MultimodalModel(schema, n_classes=2, regression=False, embedding_dim=EMB)
    inputs = {
        "labs": torch.randn(3, 12), "notes": ["a note"] * 3,
        "scan": torch.randn(3, 3, 32, 32), "expr": torch.randn(3, 20),
    }
    assert model(inputs).shape == (3, 2)
    described = model.describe()
    assert set(described["modalities"]) == set(schema)
    assert described["n_parameters"] > 0


# ------------------------------------------------------ modality wiring
def _manifest() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.normal(60, 10, 20),
        "note_text": ["a much longer clinical note describing the visit " * 3] * 20,
        "scan_path": [f"images/img_{i}.png" for i in range(20)],
        "gene_001": rng.normal(0, 1, 20),
        "gene_002": rng.normal(0, 1, 20),
        "outcome": rng.integers(0, 2, 20),
    })


def test_modality_inference_separates_the_four_kinds():
    modalities = infer_modalities(_manifest(), "outcome")
    assert modalities["image"] == ["scan_path"]
    assert modalities["text"] == ["note_text"]
    assert set(modalities["genomic"]) == {"gene_001", "gene_002"}
    assert modalities["tabular"] == ["age"]


def test_explicit_columns_override_inference():
    modalities = infer_modalities(_manifest(), "outcome", genomic_cols=["age"])
    assert "age" in modalities["genomic"]
    assert "age" not in modalities.get("tabular", [])


def test_schema_accepts_arbitrary_numeric_block_names():
    """PhysioCGM passes sensor-family blocks (ecg/ppg/eda), not the four
    inferred kinds. Each must get its own encoder."""
    df = pd.DataFrame({f"c{i}": np.random.randn(10) for i in range(6)})
    schema = build_schema({"ecg": ["c0", "c1"], "ppg": ["c2", "c3"],
                           "eda": ["c4", "c5"]}, df)
    assert set(schema) == {"ecg", "ppg", "eda"}
    assert all(spec["type"] == "tabular" and spec["input_dim"] == 2
               for spec in schema.values())


def test_snp_autodetection_from_values():
    df = pd.DataFrame({"snp_a": [0, 1, 2, 1] * 5, "snp_b": [2, 0, 1, 0] * 5})
    schema = build_schema({"genomic": ["snp_a", "snp_b"]}, df, genomic_encoding="auto")
    assert schema["genomic"]["encoding"] == "snp"
    df2 = pd.DataFrame({"gene_a": np.random.randn(20) * 5})
    schema2 = build_schema({"genomic": ["gene_a"]}, df2, genomic_encoding="auto")
    assert schema2["genomic"]["encoding"] == "expression"


def test_collate_keeps_text_as_a_list_and_stacks_tensors():
    batch = [
        ({"tab": torch.randn(4), "text": "note one"}, {"tab": 1.0, "text": 1.0},
         torch.tensor(0)),
        ({"tab": torch.randn(4), "text": "note two"}, {"tab": 1.0, "text": 0.0},
         torch.tensor(1)),
    ]
    inputs, presence, targets = collate(batch)
    assert inputs["tab"].shape == (2, 4)
    assert inputs["text"] == ["note one", "note two"]
    assert presence["text"].tolist() == [1.0, 0.0]
    assert targets.tolist() == [0, 1]
