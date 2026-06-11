import csv
from pathlib import Path

from scripts import audit_free_viewing_reference_feasibility as feasibility
from scripts import create_deepgaze_reference_configs as reference_configs


def test_feasibility_schema_order_and_rows(tmp_path, monkeypatch):
    paths = _write_feasibility_inputs(tmp_path)
    monkeypatch.setattr(feasibility, "_module_available", lambda name: name == "deepgaze_pytorch")
    monkeypatch.setattr(
        feasibility,
        "_deepgaze_class_available",
        lambda class_name: class_name in {"DeepGazeIIE", "DeepGazeMSDB"},
    )

    output = tmp_path / "decision.csv"
    rows = feasibility.audit_free_viewing_reference_feasibility(
        pyproject=paths["pyproject"],
        exporter=paths["exporter"],
        config_generator=paths["config_generator"],
        deepgaze_root=paths["deepgaze_root"],
        salicon_manifest=paths["salicon_manifest"],
        cat2000_manifest=paths["cat2000_manifest"],
        output=output,
    )
    written = _read_csv(output)

    assert list(written[0]) == feasibility.FIELDNAMES
    assert [row["candidate_reference"] for row in rows] == [
        "DeepGaze MSDB",
        "DeepGaze IIE current reference",
        "comparable modern free-viewing reference",
    ]
    assert rows[0]["decision"] == "feasible_now"
    assert rows[0]["requires_download"] == "yes"
    assert rows[0]["requires_new_dependency"] == "no"
    assert "DeepGazeMSDB class available=yes" in rows[0]["local_support"]
    assert rows[1]["decision"] == "feasible_now"
    assert rows[2]["decision"] == "defer_or_document_limitation"


def test_feasibility_marks_msdb_dependency_gap(tmp_path, monkeypatch):
    paths = _write_feasibility_inputs(tmp_path)
    monkeypatch.setattr(feasibility, "_module_available", lambda name: name == "deepgaze_pytorch")
    monkeypatch.setattr(feasibility, "_deepgaze_class_available", lambda class_name: class_name == "DeepGazeIIE")

    rows = feasibility.audit_free_viewing_reference_feasibility(
        pyproject=paths["pyproject"],
        exporter=paths["exporter"],
        config_generator=paths["config_generator"],
        deepgaze_root=paths["deepgaze_root"],
        salicon_manifest=paths["salicon_manifest"],
        cat2000_manifest=paths["cat2000_manifest"],
        output=None,
    )

    msdb = rows[0]
    assert msdb["decision"] == "requires_download_or_dependency"
    assert msdb["requires_new_dependency"] == "yes"
    assert "DeepGazeMSDB class available=no" in msdb["local_support"]


def test_msdb_reference_config_uses_existing_precomputed_loader():
    dataset = reference_configs.DATASET_BY_LABEL["salicon_static2000"]

    config = reference_configs._reference_config(
        dataset,
        precomputed_root=Path("data/precomputed/deepgaze_msdb"),
        path_template="{image_id}.npy",
        npz_key=None,
        reference_name="deepgaze_msdb_reference",
        reference_label="deepgaze_msdb_precomputed",
    )

    assert config["model"]["name"] == "deepgaze_msdb_reference"
    assert config["saliency"]["method"] == "deepgaze_precomputed"
    assert config["saliency"]["root"] == "data/precomputed/deepgaze_msdb/salicon_static2000"
    assert (
        config["output"]["dir"]
        == "outputs/real_matrix_v2/salicon_static2000/deepgaze_msdb_reference_deepgaze_msdb_precomputed"
    )


def _write_feasibility_inputs(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    exporter = tmp_path / "export_deepgaze_maps.py"
    config_generator = tmp_path / "create_deepgaze_reference_configs.py"
    deepgaze_root = tmp_path / "deepgaze"
    salicon_manifest = tmp_path / "salicon.csv"
    cat2000_manifest = tmp_path / "cat2000.csv"

    pyproject.write_text("[project]\nname='hma'\n", encoding="utf-8")
    exporter.write_text("--model deepgaze_msdb\n", encoding="utf-8")
    config_generator.write_text("--datasets --reference-name\n", encoding="utf-8")
    _write_manifest(salicon_manifest, 2)
    _write_manifest(cat2000_manifest, 3)
    for label, count in [("salicon_static2000", 2), ("cat2000_static2000", 3)]:
        root = deepgaze_root / label
        root.mkdir(parents=True)
        for index in range(count):
            (root / f"{index}.npy").write_bytes(b"npy")
    return {
        "pyproject": pyproject,
        "exporter": exporter,
        "config_generator": config_generator,
        "deepgaze_root": deepgaze_root,
        "salicon_manifest": salicon_manifest,
        "cat2000_manifest": cat2000_manifest,
    }


def _write_manifest(path, count):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path"])
        writer.writeheader()
        for index in range(count):
            writer.writerow({"image_id": str(index), "image_path": f"{index}.jpg"})


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
