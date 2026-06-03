import csv

import numpy as np
import pytest
from PIL import Image

from hma.datasets.fixation_parsers import (
    load_cat2000_observer_fixations,
    load_salicon_observer_fixations,
)
from scripts.summarize_observer_controls import summarize_observer_controls


scipy_io = pytest.importorskip("scipy.io")


def test_load_salicon_observer_fixations_reads_gaze_fixations(tmp_path):
    path = tmp_path / "salicon_fixations.mat"
    gaze = np.empty((2,), dtype=object)
    gaze[0] = {"fixations": np.asarray([[1, 2], [3, 4]], dtype=np.uint16)}
    gaze[1] = {"fixations": np.asarray([[5, 6]], dtype=np.uint16)}
    scipy_io.savemat(path, {"gaze": gaze})

    observers = load_salicon_observer_fixations(path)

    assert len(observers) == 2
    assert np.allclose(observers[0], [[1, 2], [3, 4]])
    assert np.allclose(observers[1], [[5, 6]])


def test_load_cat2000_observer_fixations_reads_fixlocs_mask(tmp_path):
    path = tmp_path / "cat_fixlocs.mat"
    fix_locs = np.zeros((4, 5), dtype=np.uint8)
    fix_locs[1, 2] = 1
    fix_locs[3, 4] = 1
    scipy_io.savemat(path, {"fixLocs": fix_locs})

    observers = load_cat2000_observer_fixations(path)

    assert len(observers) == 1
    assert np.allclose(observers[0], [[2, 1], [4, 3]])


def test_summarize_observer_controls_uses_salicon_mat_fixations(tmp_path):
    root = tmp_path / "SALICON"
    image_path = root / "images" / "val" / "image_001.jpg"
    map_path = root / "maps" / "val" / "image_001.png"
    mat_path = root / "fixations" / "val" / "image_001.mat"
    image_path.parent.mkdir(parents=True)
    map_path.parent.mkdir(parents=True)
    mat_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8), mode="RGB").save(image_path)
    target = np.zeros((8, 10), dtype=np.uint8)
    target[2:5, 2:5] = 255
    Image.fromarray(target, mode="L").save(map_path)
    gaze = np.empty((2,), dtype=object)
    gaze[0] = {"fixations": np.asarray([[2, 2], [3, 3]], dtype=np.uint16)}
    gaze[1] = {"fixations": np.asarray([[4, 4], [5, 5]], dtype=np.uint16)}
    scipy_io.savemat(mat_path, {"gaze": gaze})

    manifest = tmp_path / "salicon_manifest.csv"
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
                "image_path": "images/val/image_001.jpg",
                "fixation_map_path": "maps/val/image_001.png",
                "fixation_points_path": "fixations/val/image_001.mat",
                "split": "val",
                "width": "10",
                "height": "8",
            }
        )

    rows = summarize_observer_controls(
        manifest,
        root=root,
        dataset="salicon",
        image_size=(8, 10),
        fixation_sigma=1.0,
    )

    assert len(rows) == 2
    assert {row["control_type"] for row in rows} == {"leave_one_observer_out"}
    assert all(np.isfinite(row["inter_observer_nss"]) for row in rows)


def test_summarize_observer_controls_scales_inline_fixations(tmp_path):
    manifest = tmp_path / "coco_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "fixation_points",
                "subject_id",
                "trial_id",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "image_001",
                "fixation_points": "[[100.0, 50.0]]",
                "subject_id": "S1",
                "trial_id": "T1",
                "width": "200",
                "height": "100",
            }
        )
        writer.writerow(
            {
                "image_id": "image_001",
                "fixation_points": "[[120.0, 50.0]]",
                "subject_id": "S2",
                "trial_id": "T2",
                "width": "200",
                "height": "100",
            }
        )

    rows = summarize_observer_controls(
        manifest,
        dataset="coco_search18",
        image_size=(10, 20),
        fixation_sigma=1.0,
    )

    assert len(rows) == 2
    assert all(np.isfinite(row["inter_observer_nss"]) for row in rows)
