import csv

from scripts.audit_transformer_relevance_control import (
    EXPECTED_DATASETS,
    EXPECTED_FAMILY,
    EXPECTED_METHOD,
    EXPECTED_METRICS,
    EXPECTED_MODELS,
    FIELDNAMES,
    audit_transformer_relevance_control,
    build_transformer_relevance_control_audit,
)


def test_transformer_relevance_control_audit_passes_scoped_rows(tmp_path):
    aggregate = tmp_path / "results.csv"
    output = tmp_path / "audit.csv"
    _write_csv(aggregate, _valid_rows())

    rows = audit_transformer_relevance_control(aggregate=aggregate, output=output)
    written = _read_csv(output)

    assert list(written[0]) == FIELDNAMES
    assert all(row["status"] == "pass" for row in rows[:-1])
    assert rows[-1]["check"] == "evidence_decision"
    assert rows[-1]["status"] == "accepted_evidence_ready"


def test_transformer_relevance_control_audit_fails_wrong_dataset():
    rows = build_transformer_relevance_control_audit(
        [
            *_valid_rows(),
            _row(
                dataset="coco_search18_static2000",
                model="vit_small_patch14_dinov2",
                metric="nss",
            ),
        ],
        aggregate_artifact="results.csv",
    )

    dataset = _by_check(rows, "dataset_scope")
    decision = _by_check(rows, "evidence_decision")

    assert dataset["status"] == "fail"
    assert "coco_search18_static2000" in dataset["detail"]
    assert decision["status"] == "diagnostic_or_incomplete"


def test_transformer_relevance_control_audit_fails_wrong_family():
    bad_rows = [
        {
            **row,
            "saliency_family": "internal_routing",
        }
        for row in _valid_rows()
    ]

    rows = build_transformer_relevance_control_audit(
        bad_rows,
        aggregate_artifact="results.csv",
    )

    family = _by_check(rows, "saliency_family_label")
    decision = _by_check(rows, "evidence_decision")

    assert family["status"] == "fail"
    assert "internal_routing" in family["observed"]
    assert "must not be collapsed" in family["detail"]
    assert decision["status"] == "diagnostic_or_incomplete"


def test_transformer_relevance_control_audit_fails_missing_model():
    missing_model = "deit_small_patch16_224"
    partial_rows = [
        row for row in _valid_rows() if row["model"] != missing_model
    ]

    rows = build_transformer_relevance_control_audit(
        partial_rows,
        aggregate_artifact="results.csv",
    )

    model = _by_check(rows, "model_scope")
    cells = _by_check(rows, "expected_cell_coverage")
    decision = _by_check(rows, "evidence_decision")

    assert model["status"] == "fail"
    assert missing_model in model["detail"]
    assert cells["status"] == "fail"
    assert decision["status"] == "diagnostic_or_incomplete"


def test_transformer_relevance_control_audit_marks_missing_artifact_incomplete(tmp_path):
    rows = audit_transformer_relevance_control(
        aggregate=tmp_path / "missing.csv",
        output=None,
    )

    artifact = _by_check(rows, "aggregate_artifact_exists")
    decision = _by_check(rows, "evidence_decision")

    assert artifact["status"] == "fail"
    assert "missing or empty" in artifact["detail"]
    assert decision["status"] == "diagnostic_or_incomplete"


def _valid_rows():
    return [
        _row(dataset=dataset, model=model, metric=metric)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
        for metric in EXPECTED_METRICS
    ]


def _row(dataset, model, metric):
    return {
        "dataset": dataset,
        "model": model,
        "saliency_method": EXPECTED_METHOD,
        "saliency_family": EXPECTED_FAMILY,
        "fixation_protocol": "points",
        "metric": metric,
        "n": "2000",
        "mean": "0.5",
        "std": "0.1",
        "stderr": "0.01",
        "ci95_low": "0.48",
        "ci95_high": "0.52",
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _by_check(rows, check):
    return next(row for row in rows if row["check"] == check)
