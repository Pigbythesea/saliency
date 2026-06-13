import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hma.external.artifacts import (
    ExternalArtifactWriter,
    load_external_features,
    load_external_features_to_memmaps,
    validate_external_artifact,
)
from hma.external.adapters import _install_legacy_torch_six_compatibility
from hma.external.registry import load_external_registry
from hma.experiments.neural_alignment import run_neural_alignment
from hma.utils.config import load_yaml, save_yaml
from scripts.create_paper1_matrix_v2_configs import create_configs
from scripts.create_paper1_matrix_v2_behavior_configs import create_behavior_configs
from scripts.create_paper1_matrix_v2_cluster_jobs import create_cluster_jobs
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


def test_setup_runtime_caches_are_project_local(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_external_model, "PROJECT_ROOT", tmp_path)

    environment = setup_external_model._scratch_runtime_environment()

    for name in (
        "MAMBA_ROOT_PREFIX",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "HF_HOME",
        "TMPDIR",
    ):
        path = tmp_path / Path(environment[name]).relative_to(tmp_path)
        assert path.is_dir()
        assert str(path).startswith(str(tmp_path / "external"))


def test_external_environment_creation_uses_strict_channel_priority(
    monkeypatch,
    tmp_path,
):
    commands = []
    monkeypatch.setattr(setup_external_model, "_resolve_micromamba", lambda value: value)
    monkeypatch.setattr(
        setup_external_model,
        "_run",
        lambda command: commands.append(command),
    )
    monkeypatch.setattr(
        setup_external_model,
        "_post_install_commands",
        lambda _model_id: [],
    )
    monkeypatch.setattr(
        setup_external_model,
        "_write_environment_lock",
        lambda **_kwargs: tmp_path / "lock.txt",
    )
    monkeypatch.setattr(
        setup_external_model,
        "_environment_lock_path",
        lambda _model_id: tmp_path / "lock.txt",
    )

    setup_external_model._prepare_environment(
        model={"id": "dynamicvit_deit_small_keep_0_7"},
        source_dir=tmp_path / "source",
        environment_dir=tmp_path / "environment",
        environment_manifest=tmp_path / "environment.yaml",
        micromamba="micromamba",
    )

    assert commands[0][1:4] == ["create", "--yes", "--channel-priority"]
    assert commands[0][4] == "strict"
    assert commands[1][-3:] == [
        "python",
        "-c",
        (
            "import torch; "
            "assert torch.version.cuda, 'PyTorch was installed without CUDA support'; "
            "print(f'torch={torch.__version__} cuda={torch.version.cuda}')"
        ),
    ]


def test_legacy_torch_six_compatibility_restores_old_timm_symbols():
    class Legacy:
        pass

    class FakeTorch:
        _six = Legacy()

    _install_legacy_torch_six_compatibility(FakeTorch)

    assert FakeTorch._six.container_abcs.Mapping
    assert FakeTorch._six.string_classes == (str,)
    assert FakeTorch._six.inf == float("inf")


@pytest.mark.parametrize(
    "manifest_name",
    ["dynamicvit.yaml", "tome.yaml", "hat.yaml"],
)
def test_legacy_cuda_manifests_use_official_pytorch_builds(manifest_name):
    manifest = load_yaml(
        f"configs/external_models/environments/{manifest_name}"
    )

    channels = manifest["channels"]
    assert channels.index("defaults") < channels.index("conda-forge")
    dependencies = manifest["dependencies"]
    assert "pytorch::pytorch=1.13.1" in dependencies
    assert "pytorch::torchvision=0.14.1" in dependencies
    assert "pytorch::pytorch-cuda=11.7" in dependencies
    assert "defaults::mkl=2024.0" in dependencies


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
    cached_ids, cached, _ = load_external_features_to_memmaps(
        tmp_path / "artifact",
        layers=["blocks.0"],
        storage_dir=tmp_path / "memmaps",
    )
    assert cached_ids == mapped_ids
    assert np.array_equal(cached["blocks.0"], mapped["blocks.0"])
    assert (tmp_path / "memmaps" / "memmap_index.json").is_file()


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


def test_resource_only_artifact_omits_features_but_preserves_routing(tmp_path):
    writer = ExternalArtifactWriter(
        tmp_path / "artifact",
        model_id="fake_external",
        provenance=_provenance(),
        expected_mechanism_outputs=["prediction_masks"],
        artifact_scope="resource_only",
    )
    writer.write_batch(
        image_ids=["a", "b"],
        features={},
        logits=np.ones((2, 5), dtype=np.float32),
        resource_allocation={
            "prediction_masks.stage_0": np.ones((2, 196), dtype=np.uint8)
        },
    )
    manifest_path = writer.finalize()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_scope"] == "resource_only"
    assert manifest["features"] == {}
    assert "prediction_masks.stage_0" in manifest["resource_allocation"]


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
                "pca_cache_dir": str(tmp_path / "pca_cache"),
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
    cached_result = run_neural_alignment(config_path)

    metadata = json.loads(
        (tmp_path / "output" / "metadata.json").read_text(encoding="utf-8")
    )
    assert result["num_items"] == 12
    assert metadata["model_backend"] == "external_artifact"
    assert metadata["external_artifact_schema"] == "hma.external.artifact.v1"
    assert len(result["geometry_rows"]) == 2
    assert result["score_rows"] == cached_result["score_rows"]
    assert result["geometry_rows"] == cached_result["geometry_rows"]
    reduction = json.loads(
        (tmp_path / "output" / "feature_reduction_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert reduction["layers"][0]["cache_hit"] is True


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
    full_config = next(
        path
        for path in generated
        if "full" in path.parts and path.name == "deit_small_static_v1_full.yaml"
    )
    full_payload = load_yaml(full_config)
    assert full_payload["external_artifact"]["feature_cache_dir"].endswith(
        "deit_small_static/raw_features"
    )
    assert full_payload["neural"]["pca_cache_dir"].endswith(
        "deit_small_static/pca"
    )

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
        "cluster_smoke_pending"
    }
    assert "Cluster-readiness implementation is complete" in next_run
    assert "run_paper1_matrix_v2_scientific64.py" not in next_run


def test_cluster_jobs_match_observed_jhu_partitions(tmp_path):
    paths = create_cluster_jobs(tmp_path / "cluster")
    names = {path.name for path in paths}

    assert len(paths) == 10
    assert {"submit_smoke.sh", "submit_full.sh"} <= names
    neural = (tmp_path / "cluster" / "full_neural_analysis.sbatch").read_text(
        encoding="utf-8"
    )
    behavior = (tmp_path / "cluster" / "full_behavior_exports.sbatch").read_text(
        encoding="utf-8"
    )
    submit = (tmp_path / "cluster" / "submit_full.sh").read_text(encoding="utf-8")
    assert "#SBATCH --partition=cpu" in neural
    assert "#SBATCH --cpus-per-task=32" in neural
    assert "#SBATCH --mem=120G" in neural
    assert "#SBATCH --partition=l40s" in behavior
    assert "#SBATCH --cpus-per-task=14" in behavior
    assert "#SBATCH --gres=gpu:l40s:1" in behavior
    assert "--artifact-scope resource_only" in behavior
    assert "--dependency=afterok:$NEURAL_EXPORT" in submit
