import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from hma.external.adapters import _torch_load
from hma.behavioral.latent_fixation import (
    load_fixation_dataset_bundle,
    run_latent_fixation_encoding,
)
from hma.external.artifacts import ExternalArtifactWriter
from hma.saliency import EmpiricalSpatialPrior
from hma.utils.config import load_yaml
from scripts import create_paper1_clean_cluster_jobs as jobgen
from scripts import profile_external_model_efficiency as profiler
from scripts import run_paper1_clean_geometry as geometry_runner
from scripts import run_paper1_clean_cell_job as runner


def test_clean_external_command_uses_model_conda_prefix(tmp_path, monkeypatch):
    env_root = tmp_path / "external" / "environments"
    env_dir = env_root / "dynamicvit_deit_small_keep_0_7"
    (env_dir / "conda-meta").mkdir(parents=True)
    (env_dir / "conda-meta" / "history").write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "EXTERNAL_ENV_ROOT", env_root.resolve())
    monkeypatch.setenv("CONDA_EXE", "/scratch/tshu2/zzhan330/miniconda3/bin/conda")
    monkeypatch.setenv("HMA_EXTERNAL_ENV_MODE", "prefix")

    command = runner.external_model_command(
        "dynamicvit_deit_small_keep_0_7",
        ["scripts/run_external_model.py", "--model", "dynamicvit_deit_small_keep_0_7"],
    )

    assert command[:6] == [
        "/scratch/tshu2/zzhan330/miniconda3/bin/conda",
        "run",
        "--no-capture-output",
        "-p",
        str(env_dir.resolve()),
        "python",
    ]


def test_clean_external_command_can_use_active_hma_env(monkeypatch):
    monkeypatch.setenv("HMA_EXTERNAL_ENV_MODE", "active")

    command = runner.external_model_command(
        "resnet50",
        ["scripts/run_external_model.py", "--model", "resnet50"],
    )

    assert command == [
        sys.executable,
        "scripts/run_external_model.py",
        "--model",
        "resnet50",
    ]


def test_clean_external_command_rejects_missing_or_escaped_env(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXTERNAL_ENV_ROOT", (tmp_path / "envs").resolve())
    monkeypatch.setenv("HMA_EXTERNAL_ENV_MODE", "prefix")

    with pytest.raises(runner.CleanCellJobError, match="Invalid runtime_model_id"):
        runner.external_model_command("../hma", ["script.py"])

    with pytest.raises(runner.CleanCellJobError, match="not ready"):
        runner.external_model_command("missing_model", ["script.py"])


def test_deepgaze_msdb_clean_exports_use_single_image_batch():
    assert jobgen.export_batch_size("deepgaze_msdb") == "1"
    assert jobgen.export_batch_size("resnet50") == "16"


def test_clean_submit_script_pipelines_neural_waves_through_cleanup():
    script = jobgen.submit_script(
        mode="full",
        include_materialize=True,
        behavior_count=5,
        neural_count=25,
    )

    assert "clean_behavior_cells.sbatch" in script
    assert "clean_behavior_latent_analysis.sbatch" in script
    assert "clean_behavior_latent_cleanup.sbatch" in script
    assert "PAPER1_BEHAVIOR_EXPORT_CONCURRENCY" in script
    assert "BEHAVIOR=\"$BEHAVIOR_CLEANUP\"" in script
    assert 'NEURAL_WAVE_SIZE="${PAPER1_NEURAL_WAVE_SIZE:-48}"' in script
    assert 'NEURAL_EXPORT_CONCURRENCY="${PAPER1_NEURAL_EXPORT_CONCURRENCY:-4}"' in script
    assert "EFFICIENCY_GPU_CONCURRENCY" not in script
    assert "BEHAVIOR_GPU_CONCURRENCY" not in script
    assert "clean_neural_cleanup.sbatch" in script
    assert "--array=${START}-${END}%${NEURAL_EXPORT_CONCURRENCY}" in script
    assert "--dependency=afterany:$PREVIOUS_NEURAL_CLEANUP" in script
    assert "--dependency=aftercorr:$NEURAL_EXPORT" in script
    assert "--dependency=aftercorr:$NEURAL_ANALYSIS" in script
    assert "GEOMETRY=$(sbatch --parsable --dependency=afterany:$PREVIOUS_NEURAL_CLEANUP" in script
    assert "MATERIALIZE=$(sbatch --parsable --dependency=afterok:$BEHAVIOR:$GEOMETRY:$EFFICIENCY" in script


def test_primary_behavioral_latent_fixation_trains_and_scores(tmp_path):
    bundle = load_fixation_dataset_bundle(
        "dummy_saliency",
        {
            "name": "dummy_saliency",
            "split": "smoke",
            "num_items": 6,
            "map_shape": [8, 8],
            "num_fixation_points": 4,
            "artifact_key": "image_id",
            "regime": "free_viewing",
            "behavioral_object": "latent_encoded_human_fixation_density",
            "seed": 7,
        },
        target_size=(8, 8),
    )
    features = np.stack(
        [
            np.asarray(target, dtype=np.float32).reshape(-1)[:16]
            for target in bundle.targets
        ],
        axis=0,
    )
    artifact_dir = tmp_path / "outputs" / "paper1_publication_v0" / "behavioral_latent_fixation" / "external" / "dummy" / "model"
    writer = ExternalArtifactWriter(
        artifact_dir,
        model_id="dummy_latent_model",
        provenance={
            "model_id": "dummy_latent_model",
            "repository": "test",
            "repository_commit": "abc123",
            "environment_hash": "env123",
            "checkpoint_hash": "ckpt123",
            "seed": 123,
            "hardware": {"device": "cpu"},
            "preprocessing": {"kind": "test"},
        },
        expected_mechanism_outputs=[],
        artifact_scope="full",
    )
    writer.write_batch(
        image_ids=bundle.artifact_ids,
        features={"layer": features},
    )
    writer.finalize()

    aggregate_rows, image_rows, selection, reduction = run_latent_fixation_encoding(
        bundle=bundle,
        artifact_dir=artifact_dir,
        model_id="dummy_latent_model",
        layers=["layer"],
        ridge_alphas=[0.1, 1.0],
        pca_components=4,
        train_fraction=0.75,
        validation_fraction_of_train=0.25,
        seed=123,
    )

    primary = [
        row for row in aggregate_rows
        if row["metric"] == "latent_fixation_information_gain"
    ]
    assert len(primary) == 1
    assert primary[0]["primary_behavioral_score"] == "yes"
    assert np.isfinite(float(primary[0]["mean"]))
    assert image_rows
    assert {row["split_role"] for row in image_rows} == {"test"}
    assert selection["selected_layer"] == "layer"
    assert reduction[0]["pca_fit_scope"] in {
        "training_only",
        "not_applied_feature_dim_within_limit",
    }


def test_cluster_jobs_default_external_models_to_active_hma_env():
    script = jobgen.array_job(
        name="p1c_smoke_nexp",
        partition="l40s",
        time="00:30:00",
        cpus=2,
        memory="8G",
        gpu=True,
        count=1,
        command="python scripts/run_paper1_clean_cell_job.py --kind neural_export",
    )

    assert 'export HMA_EXTERNAL_ENV_MODE="${HMA_EXTERNAL_ENV_MODE:-active}"' in script


def test_neural_only_submit_script_does_not_submit_behavior_or_efficiency():
    script = jobgen.neural_only_submit_script(
        mode="full",
        include_neural_materialize=True,
        neural_count=25,
    )

    assert "clean_neural_exports.sbatch" in script
    assert "clean_neural_cleanup.sbatch" in script
    assert "clean_neural_materialize.sbatch" in script
    assert 'NEURAL_EXPORT_CONCURRENCY="${PAPER1_NEURAL_EXPORT_CONCURRENCY:-4}"' in script
    assert "clean_behavior_cells.sbatch" not in script
    assert "clean_efficiency_cells.sbatch" not in script
    assert "clean_materialize_axes.sbatch" not in script


def test_no_behavior_submit_script_includes_efficiency_but_not_behavior():
    script = jobgen.no_behavior_submit_script(
        mode="full",
        include_no_behavior_materialize=True,
        neural_count=25,
    )

    assert "clean_neural_exports.sbatch" in script
    assert "clean_neural_cleanup.sbatch" in script
    assert 'NEURAL_EXPORT_CONCURRENCY="${PAPER1_NEURAL_EXPORT_CONCURRENCY:-4}"' in script
    assert "clean_efficiency_cells.sbatch" in script
    assert "EFFICIENCY_GPU_CONCURRENCY" not in script
    assert "clean_no_behavior_materialize.sbatch" in script
    assert "--dependency=afterok:$GEOMETRY:$EFFICIENCY" in script
    assert "clean_behavior_cells.sbatch" not in script
    assert "clean_materialize_axes.sbatch" not in script


def test_publication_geometry_protocol_uses_aggregate_stability_not_cell_bootstrap():
    config = load_yaml("configs/paper1_latent_neural_matrix.yaml")
    geometry = config["geometry"]

    assert geometry["protocol"] == "primary_deterministic_seed_stability_v1"
    assert geometry["subset_sizes"] == [1024]
    assert geometry["subset_seeds"] == [123, 456, 789]
    assert geometry["image_resampling"]["resamples"] == 0
    assert geometry["image_resampling"]["method"] == "aggregate_level_only"
    assert geometry["aggregate_uncertainty"]["outputs"] == [
        "geometry_method_agreement",
        "geometry_seed_stability",
    ]


def test_geometry_materializer_builds_method_agreement_and_seed_stability():
    rows = [
        {
            "model_id": "model",
            "subject_id": "subj01",
            "roi": "V1",
            "stream": "early",
            "roi_class": "retinotopic",
            "layer": "layer1",
            "geometry_method": "debiased_linear_cka",
            "score": "0.25",
            "control_type": "observed",
        },
        {
            "model_id": "model",
            "subject_id": "subj01",
            "roi": "V1",
            "stream": "early",
            "roi_class": "retinotopic",
            "layer": "layer1",
            "geometry_method": "subset_rsa",
            "score": "0.10",
            "subset_size": "1024",
            "subset_seed": "123",
            "control_type": "observed",
        },
        {
            "model_id": "model",
            "subject_id": "subj01",
            "roi": "V1",
            "stream": "early",
            "roi_class": "retinotopic",
            "layer": "layer1",
            "geometry_method": "subset_rsa",
            "score": "0.20",
            "subset_size": "1024",
            "subset_seed": "456",
            "control_type": "observed",
        },
        {
            "model_id": "model",
            "subject_id": "subj01",
            "roi": "V1",
            "stream": "early",
            "roi_class": "retinotopic",
            "layer": "layer1",
            "geometry_method": "subset_rsa",
            "score": "0.99",
            "subset_size": "1024",
            "subset_seed": "123",
            "control_type": "response_permutation",
        },
    ]

    agreement = geometry_runner.build_geometry_method_agreement_rows(rows)
    stability = geometry_runner.build_geometry_seed_stability_rows(rows)

    assert len(agreement) == 1
    assert agreement[0]["cka_score"] == pytest.approx(0.25)
    assert agreement[0]["subset_rsa_mean_score"] == pytest.approx(0.15)
    assert agreement[0]["subset_rsa_seed_count"] == 2
    assert agreement[0]["absolute_score_delta"] == pytest.approx(0.10)
    assert len(stability) == 1
    assert stability[0]["seed_count"] == 2
    assert stability[0]["mean_score"] == pytest.approx(0.15)
    assert stability[0]["score_range"] == pytest.approx(0.10)
    assert stability[0]["stability_status"] == "ok"


def test_clean_efficiency_runner_allows_explicit_failed_profile(monkeypatch):
    commands = []

    def fake_external_model_command(runtime_model_id, args):
        return ["python", *args]

    def fake_run(command):
        commands.append(command)

    monkeypatch.setattr(runner, "external_model_command", fake_external_model_command)
    monkeypatch.setattr(runner, "run", fake_run)

    runner.run_efficiency(
        {
            "runtime_model_id": "scandiff",
            "manifest": "data/manifests/nsd_algonauts_manifest.csv",
            "image_root": "data/raw/nsd_algonauts",
            "subject_id": "subj01",
            "roi": "V1",
            "device": "cuda",
            "output_json": "outputs/paper1_publication_v0/efficiency/cells/scandiff_freeview/efficiency.json",
        }
    )

    assert len(commands) == 1
    assert "--allow-failure-profile" in commands[0]


def test_efficiency_profiler_writes_failed_profile_payload(tmp_path, monkeypatch):
    output = tmp_path / "outputs" / "paper1_publication_v0" / "efficiency" / "cells" / "scandiff_freeview" / "efficiency.json"
    manifest = tmp_path / "manifest.csv"
    image_root = tmp_path / "images"
    manifest.write_text("image_id,image_path,split\n", encoding="utf-8")
    image_root.mkdir()

    def fail_profile(*args, **kwargs):
        raise RuntimeError("missing hydra")

    monkeypatch.setattr(profiler, "profile_model", fail_profile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_external_model_efficiency.py",
            "--model",
            "scandiff",
            "--manifest",
            str(manifest),
            "--image-root",
            str(image_root),
            "--output",
            str(output),
            "--allow-failure-profile",
        ],
    )

    assert profiler.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model_id"] == "scandiff"
    assert payload["publication_model_id"] == "scandiff_freeview"
    assert payload["profile_status"] == "profile_failed"
    assert payload["failure_type"] == "RuntimeError"
    assert payload["failure_message"] == "missing hydra"
    assert payload["resource_summary"]["profile_failure"]["status"] == "profile_failed"


def test_neural_cleanup_preserves_provenance_and_removes_intermediates(
    tmp_path,
    monkeypatch,
):
    publication_root = tmp_path / "outputs" / "paper1_publication_v0"
    output_dir = publication_root / "neural_encoding" / "cells" / "model" / "subj01_V1"
    artifact_dir = publication_root / "external" / "neural" / "model__subj01__V1"
    raw_cache = publication_root / "neural_encoding" / "cache" / "model" / "subj01_V1" / "raw_features"
    pca_cache = publication_root / "neural_encoding" / "cache" / "model" / "subj01_V1" / "pca"
    config_path = (
        publication_root
        / "preflight"
        / "runtime_configs"
        / "full"
        / "neural"
        / "model__subj01__V1.yaml"
    )
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    raw_cache.mkdir(parents=True)
    pca_cache.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    for name in (
        "feature_reduction_metadata.json",
        "encoding_scores.csv",
        "encoding_target_scores.csv",
        "geometry_scores.csv",
    ):
        (output_dir / name).write_text("ok\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_backend": "external_artifact",
                "external_artifact_schema": "hma.external.artifact.v1",
                "external_artifact_provenance": {"model_id": "model"},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "image_ids.json").write_text('["image_001"]\n', encoding="utf-8")
    (artifact_dir / "efficiency.json").write_text("{}\n", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "hma.external.artifact.v1",
                "image_ids_file": "image_ids.json",
                "efficiency_file": "efficiency.json",
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "chunk.npz").write_text("large", encoding="utf-8")
    (raw_cache / "features.dat").write_text("raw", encoding="utf-8")
    (pca_cache / "pca.npy").write_text("pca", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "external_artifact:",
                f"  feature_cache_dir: {raw_cache.as_posix()}",
                "neural:",
                f"  pca_cache_dir: {pca_cache.as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "PUBLICATION_ROOT", publication_root.resolve())

    row = {
        "model_id": "model",
        "subject_id": "subj01",
        "roi": "V1",
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "config_path": str(config_path),
    }
    runner.cleanup_neural_intermediates(row)

    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "external_artifact_manifest.json").is_file()
    assert (output_dir / "external_artifact_image_ids.json").is_file()
    assert (output_dir / "intermediate_cleanup.json").is_file()
    assert not artifact_dir.exists()
    assert not raw_cache.exists()
    assert not pca_cache.exists()


def test_neural_cleanup_refuses_to_delete_without_final_outputs(tmp_path, monkeypatch):
    publication_root = tmp_path / "outputs" / "paper1_publication_v0"
    output_dir = publication_root / "neural_encoding" / "cells" / "model" / "subj01_V1"
    artifact_dir = publication_root / "external" / "neural" / "model__subj01__V1"
    config_path = publication_root / "preflight" / "runtime_configs" / "full" / "neural" / "model.yaml"
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("external_artifact: {}\nneural: {}\n", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        '{"schema_version": "hma.external.artifact.v1"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "PUBLICATION_ROOT", publication_root.resolve())

    with pytest.raises(runner.CleanCellJobError, match="final neural outputs are missing"):
        runner.cleanup_neural_intermediates(
            {
                "output_dir": str(output_dir),
                "artifact_dir": str(artifact_dir),
                "config_path": str(config_path),
            }
        )

    assert artifact_dir.exists()


def test_empirical_spatial_prior_reads_salicon_fixation_points_path(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    root = tmp_path / "data" / "raw" / "SALICON"
    fix_path = root / "fixations" / "train" / "image_001.mat"
    fix_path.parent.mkdir(parents=True)
    gaze = np.empty((2,), dtype=object)
    gaze[0] = {"fixations": np.asarray([[2, 2], [3, 2]], dtype=np.uint16)}
    gaze[1] = {"fixations": np.asarray([[4, 2]], dtype=np.uint16)}
    scipy_io.savemat(fix_path, {"gaze": gaze})

    manifest = tmp_path / "data" / "manifests" / "salicon_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_path",
                "fixation_map_path",
                "fixation_points_path",
                "split",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "image_001",
                "image_path": "images/train/image_001.jpg",
                "fixation_map_path": "maps/train/image_001.png",
                "fixation_points_path": "fixations/train/image_001.mat",
                "split": "train",
                "width": "10",
                "height": "8",
            }
        )

    prior = EmpiricalSpatialPrior.from_manifest(
        manifest,
        image_size=(8, 10),
        fixation_sigma=1.0,
    )

    y, x = np.unravel_index(np.argmax(prior.map), prior.map.shape)
    assert x in {2, 3, 4}
    assert y == 2


def test_torch_load_disables_weights_only_when_supported(tmp_path):
    class FakeTorch:
        def __init__(self):
            self.calls = []

        def load(self, path, *, map_location, weights_only=None):
            self.calls.append(
                {
                    "path": path,
                    "map_location": map_location,
                    "weights_only": weights_only,
                }
            )
            return {"state_dict": {}}

    fake = FakeTorch()
    path = tmp_path / "checkpoint.pth"

    assert _torch_load(path, fake) == {"state_dict": {}}
    assert fake.calls == [
        {
            "path": path,
            "map_location": "cpu",
            "weights_only": False,
        }
    ]


def test_torch_load_falls_back_for_old_torch_without_weights_only(tmp_path):
    class OldTorch:
        def __init__(self):
            self.calls = 0

        def load(self, path, *, map_location, weights_only=None):
            self.calls += 1
            if weights_only is not None:
                raise TypeError("load() got an unexpected keyword argument 'weights_only'")
            return {"state_dict": {}}

    fake = OldTorch()
    path = tmp_path / "checkpoint.pth"

    assert _torch_load(path, fake) == {"state_dict": {}}
    assert fake.calls == 2
