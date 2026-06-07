import csv

import numpy as np
import pytest

from scripts import create_nsd_noise_ceiling_manifest as converter


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_noise_ceiling_from_ncsnr_uses_trial_count_formula():
    values = converter._noise_ceiling_from_ncsnr(
        np.array([0.0, 1.0, 2.0]),
        n_trials=3,
    )

    assert np.allclose(values, [0.0, 0.75, 12.0 / 13.0])


def test_create_nsd_noise_ceiling_manifest_writes_roi_sidecars_and_manifest(
    monkeypatch,
    tmp_path,
):
    algonauts_root = tmp_path / "algonauts"
    nsd_full_root = tmp_path / "nsd_full"
    mask_dir = algonauts_root / "subj01" / "roi_masks"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "mapping_prf-visualrois.npy", {0: "Unknown", 1: "V1v", 2: "V1d", 7: "hV4"})
    np.save(mask_dir / "lh.all-vertices_fsaverage_space.npy", np.array([1, 1, 0, 1, 0, 1]))
    np.save(mask_dir / "rh.all-vertices_fsaverage_space.npy", np.array([0, 1, 1, 1, 1, 0]))
    np.save(mask_dir / "lh.prf-visualrois_challenge_space.npy", np.array([1, 2, 7, 0]))
    np.save(mask_dir / "rh.prf-visualrois_challenge_space.npy", np.array([2, 7, 1, 0]))

    response_root = algonauts_root / "subj01" / "responses"
    (response_root / "V1").mkdir(parents=True)
    (response_root / "hV4").mkdir(parents=True)
    np.save(response_root / "V1" / "train-0001_nsd-00001.npy", np.zeros(4, dtype=np.float32))
    np.save(response_root / "hV4" / "train-0001_nsd-00001.npy", np.zeros(2, dtype=np.float32))

    def fake_mgh(path):
        if path.name.startswith("lh"):
            return np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        return np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

    monkeypatch.setattr(converter, "_load_mgh_vector", fake_mgh)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_path",
                "split",
                "subject_id",
                "roi",
                "roi_response_path",
                "roi_responses",
            ],
        )
        writer.writeheader()
        for roi in ["V1", "hV4"]:
            writer.writerow(
                {
                    "image_id": "train-0001_nsd-00001",
                    "image_path": "subj01/training_split/training_images/train-0001_nsd-00001.png",
                    "split": "train",
                    "subject_id": "subj01",
                    "roi": roi,
                    "roi_response_path": f"subj01/responses/{roi}/train-0001_nsd-00001.npy",
                    "roi_responses": "",
                }
            )

    result = converter.create_nsd_noise_ceiling_manifest(
        algonauts_root=algonauts_root,
        nsd_full_root=nsd_full_root,
        manifest_path=manifest,
        roi_names=["V1", "hV4"],
        n_trials=3,
    )

    v1 = np.load(algonauts_root / "subj01" / "noise_ceilings" / "V1.npy")
    hv4 = np.load(algonauts_root / "subj01" / "noise_ceilings" / "hV4.npy")
    assert v1.shape == (4,)
    assert hv4.shape == (2,)
    assert result["rows"] == 2

    rows = _read_csv(manifest)
    assert rows[0]["noise_ceiling_path"] == "subj01/noise_ceilings/V1.npy"
    assert rows[0]["noise_ceiling_source"] == "nsd_ncsnr_mgh_n_trials_3"
    assert rows[1]["noise_ceiling_path"] == "subj01/noise_ceilings/hV4.npy"
    assert (algonauts_root / "subj01" / "noise_ceilings" / "noise_ceiling_summary.csv").is_file()


def test_create_nsd_noise_ceiling_manifest_supports_stream_roi_class(
    monkeypatch,
    tmp_path,
):
    algonauts_root = tmp_path / "algonauts"
    nsd_full_root = tmp_path / "nsd_full"
    mask_dir = algonauts_root / "subj01" / "roi_masks"
    mask_dir.mkdir(parents=True)
    np.save(
        mask_dir / "mapping_streams.npy",
        {0: "Unknown", 2: "midventral", 3: "midlateral", 5: "ventral"},
    )
    np.save(mask_dir / "lh.all-vertices_fsaverage_space.npy", np.array([1, 1, 1, 0, 1]))
    np.save(mask_dir / "rh.all-vertices_fsaverage_space.npy", np.array([1, 0, 1, 1, 1]))
    np.save(mask_dir / "lh.streams_challenge_space.npy", np.array([2, 3, 5, 2]))
    np.save(mask_dir / "rh.streams_challenge_space.npy", np.array([0, 2, 5, 3]))

    response_root = algonauts_root / "subj01" / "responses"
    (response_root / "midventral").mkdir(parents=True)
    (response_root / "ventral").mkdir(parents=True)
    np.save(response_root / "midventral" / "train-0001_nsd-00001.npy", np.zeros(3, dtype=np.float32))
    np.save(response_root / "ventral" / "train-0001_nsd-00001.npy", np.zeros(2, dtype=np.float32))

    def fake_mgh(path):
        if path.name.startswith("lh"):
            return np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        return np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    monkeypatch.setattr(converter, "_load_mgh_vector", fake_mgh)
    manifest = tmp_path / "streams_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_path",
                "split",
                "subject_id",
                "roi",
                "roi_response_path",
                "roi_responses",
            ],
        )
        writer.writeheader()
        for roi in ["midventral", "ventral"]:
            writer.writerow(
                {
                    "image_id": "train-0001_nsd-00001",
                    "image_path": "subj01/training_split/training_images/train-0001_nsd-00001.png",
                    "split": "train",
                    "subject_id": "subj01",
                    "roi": roi,
                    "roi_response_path": f"subj01/responses/{roi}/train-0001_nsd-00001.npy",
                    "roi_responses": "",
                }
            )

    result = converter.create_nsd_noise_ceiling_manifest(
        algonauts_root=algonauts_root,
        nsd_full_root=nsd_full_root,
        manifest_path=manifest,
        roi_class="streams",
        roi_names=["midventral", "ventral"],
        n_trials=3,
    )

    midventral = np.load(algonauts_root / "subj01" / "noise_ceilings" / "midventral.npy")
    ventral = np.load(algonauts_root / "subj01" / "noise_ceilings" / "ventral.npy")
    assert midventral.shape == (3,)
    assert ventral.shape == (2,)
    assert result["roi_class"] == "streams"
    rows = _read_csv(manifest)
    assert rows[0]["noise_ceiling_path"] == "subj01/noise_ceilings/midventral.npy"
    assert rows[0]["noise_ceiling_source"] == "nsd_ncsnr_mgh_n_trials_3"
    assert rows[1]["noise_ceiling_path"] == "subj01/noise_ceilings/ventral.npy"


def test_create_nsd_noise_ceiling_manifest_rejects_response_dimension_mismatch(
    monkeypatch,
    tmp_path,
):
    algonauts_root = tmp_path / "algonauts"
    nsd_full_root = tmp_path / "nsd_full"
    mask_dir = algonauts_root / "subj01" / "roi_masks"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "mapping_streams.npy", {0: "Unknown", 2: "midventral"})
    np.save(mask_dir / "lh.all-vertices_fsaverage_space.npy", np.array([1, 1, 1]))
    np.save(mask_dir / "rh.all-vertices_fsaverage_space.npy", np.array([1, 1, 1]))
    np.save(mask_dir / "lh.streams_challenge_space.npy", np.array([2, 0, 0]))
    np.save(mask_dir / "rh.streams_challenge_space.npy", np.array([2, 0, 0]))

    response_dir = algonauts_root / "subj01" / "responses" / "midventral"
    response_dir.mkdir(parents=True)
    np.save(response_dir / "train-0001_nsd-00001.npy", np.zeros(1, dtype=np.float32))

    monkeypatch.setattr(
        converter,
        "_load_mgh_vector",
        lambda path: np.array([1.0, 2.0, 3.0]),
    )
    manifest = tmp_path / "streams_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_path",
                "split",
                "subject_id",
                "roi",
                "roi_response_path",
                "roi_responses",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "train-0001_nsd-00001",
                "image_path": "subj01/training_split/training_images/train-0001_nsd-00001.png",
                "split": "train",
                "subject_id": "subj01",
                "roi": "midventral",
                "roi_response_path": "subj01/responses/midventral/train-0001_nsd-00001.npy",
                "roi_responses": "",
            }
        )

    with pytest.raises(ValueError, match="noise ceiling has 2 targets"):
        converter.create_nsd_noise_ceiling_manifest(
            algonauts_root=algonauts_root,
            nsd_full_root=nsd_full_root,
            manifest_path=manifest,
            roi_class="streams",
            roi_names=["midventral"],
            n_trials=3,
        )
