import json
import hashlib
import subprocess
import sys

import numpy as np
import pytest

from hma.external.artifacts import (
    ExternalArtifactWriter,
    load_external_features,
    load_external_features_to_memmaps,
    validate_external_artifact,
)
from hma.external.registry import load_external_registry
from hma.experiments.neural_alignment import run_neural_alignment
from hma.utils.config import save_yaml
from scripts.create_paper1_matrix_v2_configs import create_configs
from scripts.create_paper1_matrix_v2_behavior_configs import create_behavior_configs
from scripts.export_external_routing_maps import export_routing_maps
from scripts import setup_external_model
from scripts.audit_paper1_matrix_v2 import (
    _adaptive_matrix_rows,
    _next_run_markdown,
)
from scripts.run_paper1_matrix_v2_scientific64 import build_jobs


def test_setup_module_loads_without_numpy_or_pillow():
    code = """
import builtins
import runpy

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy.") or name == "PIL" or name.startswith("PIL."):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
namespace = runpy.run_path(
    "scripts/setup_external_model.py",
    run_name="hma_setup_bootstrap_probe",
)
report = namespace["build_installation_report"]("deit_small_static")
assert report["stages"]["adapter_ready"] is True
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_setup_bootstraps_pinned_micromamba_with_checksum(monkeypatch, tmp_path):
    payload = b"fake-micromamba"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(setup_external_model, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(setup_external_model.shutil, "which", lambda _value: None)
    monkeypatch.setattr(setup_external_model.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_external_model.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        setup_external_model,
        "MICROMAMBA_ASSETS",
        {"x86_64": {"url": "https://example.invalid/micromamba", "sha256": digest}},
    )

    def fake_download(_url, target):
        target.write_bytes(payload)

    monkeypatch.setattr(setup_external_model.urllib.request, "urlretrieve", fake_download)

    executable = setup_external_model._resolve_micromamba("micromamba")

    path = tmp_path / "external/tools/micromamba/2.8.1-0/micromamba"
    assert executable == str(path)
    assert path.read_bytes() == payload


def test_scientific64_runner_builds_resilient_local_job_sequence():
    jobs = build_jobs()

    assert len(jobs) == 16
    assert [job["kind"] for job in jobs[:3]] == ["export", "export", "export"]
    assert all(
        "external/tools/micromamba/2.8.1-0/micromamba"
        in " ".join(job["command"])
        for job in jobs[:3]
    )
    analysis = [job for job in jobs if job["kind"] == "analysis"]
    assert len(analysis) == 12
    assert all("--config" in job["command"] for job in analysis)
    assert jobs[-1]["kind"] == "audit"


def _provenance():
    return {
        "model_id": "fake_external",
        "repository": "https://example.invalid/fake.git",
        "repository_commit": "a" * 40,
        "environment_hash": "b" * 64,
        "checkpoint_hash": "c" * 64,
        "seed": 123,
        "hardware": {"device_name": "test"},
        "preprocessing": {"input_size": [8, 8]},
    }


def test_external_artifact_round_trip_preserves_order_and_shape(tmp_path):
    writer = ExternalArtifactWriter(
        tmp_path / "artifact",
        model_id="fake_external",
        provenance=_provenance(),
        expected_mechanism_outputs=["prediction_masks"],
    )
    writer.write_batch(
        image_ids=["a", "b"],
        features={"blocks.0": np.arange(24, dtype=np.float32).reshape(2, 3, 4)},
        logits=np.ones((2, 5), dtype=np.float32),
        resource_allocation={
            "prediction_masks.stage_0": np.ones((2, 3), dtype=np.uint8)
        },
    )
    writer.write_batch(
        image_ids=["c"],
        features={"blocks.0": np.arange(12, dtype=np.float32).reshape(1, 3, 4)},
        logits=np.ones((1, 5), dtype=np.float32),
        resource_allocation={
            "prediction_masks.stage_0": np.ones((1, 3), dtype=np.uint8)
        },
    )
    writer.set_efficiency({"parameters": 10})
    writer.finalize()

    image_ids, features, manifest = load_external_features(tmp_path / "artifact")

    assert image_ids == ["a", "b", "c"]
    assert features["blocks.0"].shape == (3, 3, 4)
    assert manifest["provenance"]["checkpoint_hash"] == "c" * 64

    mapped_ids, mapped, _ = load_external_features_to_memmaps(
        tmp_path / "artifact",
        layers=["blocks.0"],
        storage_dir=tmp_path / "memmaps",
    )
    assert mapped_ids == image_ids
    assert isinstance(mapped["blocks.0"], np.memmap)
    assert mapped["blocks.0"].shape == (3, 3, 4)


def test_external_artifact_rejects_missing_operational_output(tmp_path):
    writer = ExternalArtifactWriter(
        tmp_path / "artifact",
        model_id="fake_external",
        provenance=_provenance(),
        expected_mechanism_outputs=["token_source_assignments"],
    )
    writer.write_batch(
        image_ids=["a"],
        features={"blocks.0": np.ones((1, 2, 3), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="operational outputs"):
        writer.finalize()


def test_routing_map_export_uses_final_dynamicvit_mask(tmp_path):
    writer = ExternalArtifactWriter(
        tmp_path / "artifact",
        model_id="fake_external",
        provenance=_provenance(),
        expected_mechanism_outputs=["prediction_masks"],
    )
    stage0 = np.ones((1, 196), dtype=np.uint8)
    stage2 = np.zeros((1, 196), dtype=np.uint8)
    stage2[:, :49] = 1
    writer.write_batch(
        image_ids=["image-a"],
        features={"blocks.0": np.ones((1, 197, 2), dtype=np.float32)},
        resource_allocation={
            "prediction_masks.stage_0": stage0,
            "prediction_masks.stage_2": stage2,
        },
    )
    writer.finalize()

    metadata_path = export_routing_maps(
        tmp_path / "artifact",
        tmp_path / "maps",
        output_size=(28, 28),
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    routing_map = np.load(tmp_path / "maps" / "image-a.npy")
    assert metadata["resource_key"] == "prediction_masks.stage_2"
    assert routing_map.shape == (28, 28)
    assert routing_map.max() == pytest.approx(1.0)
    assert routing_map.min() == pytest.approx(0.0)


def test_neural_runner_imports_external_features_without_building_model(
    monkeypatch, tmp_path
):
    writer = ExternalArtifactWriter(
        tmp_path / "artifact",
        model_id="fake_external",
        provenance=_provenance(),
    )
    image_ids = [f"val_{index:04d}" for index in range(12)]
    rng = np.random.default_rng(3)
    writer.write_batch(
        image_ids=image_ids,
        features={"embedding": rng.normal(size=(12, 2, 3)).astype(np.float32)},
    )
    writer.finalize()
    config_path = tmp_path / "external_neural.yaml"
    save_yaml(
        {
            "seed": 123,
            "device": "cpu",
            "dataset": {
                "name": "dummy_static_saliency",
                "label": "external_neural",
                "num_items": 12,
                "image_shape": [3, 8, 8],
                "map_shape": [8, 8],
                "roi_response_dim": 2,
            },
            "model": {
                "name": "fake_external",
                "backend": "external_artifact",
                "pretrained": True,
            },
            "external_artifact": {"path": str(tmp_path / "artifact")},
            "neural": {
                "layers": ["embedding"],
                "response_key": "roi_responses",
                "feature_reduction": "flatten_pca",
                "pca_components": 4,
                "train_fraction": 0.75,
                "ridge_alpha": 1.0,
                "geometry": {
                    "enabled": True,
                    "methods": ["linear_cka", "subset_rsa"],
                    "subset_sizes": [6],
                    "subset_seed": 123,
                },
            },
            "output": {"dir": str(tmp_path / "output")},
        },
        config_path,
    )
    from hma.experiments import neural_alignment

    monkeypatch.setattr(
        neural_alignment,
        "build_model",
        lambda _config: pytest.fail("external artifacts must not build a core model"),
    )

    result = run_neural_alignment(config_path)

    metadata = json.loads(
        (tmp_path / "output" / "metadata.json").read_text(encoding="utf-8")
    )
    assert result["num_items"] == 12
    assert metadata["model_backend"] == "external_artifact"
    assert metadata["external_artifact_schema"] == "hma.external.artifact.v1"
    assert len(result["geometry_rows"]) == 2


def test_matrix_v2_registry_and_generated_configs_are_complete(tmp_path):
    registry = load_external_registry()
    required = {
        "deit_small_static",
        "dynamicvit_deit_small_keep_0_7",
        "tome_deit_small_r13",
        "dinov3_small_patch16",
        "siglip_base_patch16",
        "mambavision_t",
        "hiera_tiny",
        "swin_tiny",
        "hat",
        "scandiff",
    }
    assert required <= set(registry.models)

    generated = create_configs(output_root=tmp_path / "configs")

    assert len(generated) == 36
    scientific = [
        path
        for path in generated
        if "scientific64" in path.parts and "dynamicvit" in path.name
    ]
    assert len(scientific) == 4

    behavior = create_behavior_configs(output_root=tmp_path / "behavior")
    assert len(behavior) == 9


def test_matrix_v2_audit_advances_after_scientific64_acceptance():
    from hma.utils.config import load_yaml

    matrix = load_yaml("configs/paper1_matrix_v2.yaml")
    rows = _adaptive_matrix_rows(matrix)
    next_run = _next_run_markdown(matrix)

    assert len(rows) == 12
    assert {row["scientific64_status"] for row in rows} == {"passed"}
    assert {row["full_status"] for row in rows} == {
        "cluster_readiness_implementation_pending"
    }
    assert "cluster-readiness implementation" in next_run
    assert "run_paper1_matrix_v2_scientific64.py" not in next_run
