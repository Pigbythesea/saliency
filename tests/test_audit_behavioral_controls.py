import csv

from scripts.audit_behavioral_controls import (
    FIELDNAMES,
    audit_behavioral_controls,
    build_behavioral_control_gap_audit,
)


def test_behavioral_control_gap_audit_schema_order_and_statuses(tmp_path):
    behavior = tmp_path / "behavior.csv"
    observers = tmp_path / "observers.csv"
    output = tmp_path / "audit.csv"
    _write_behavior_rows(behavior)
    _write_observer_rows(observers)

    rows = audit_behavioral_controls(
        behavioral_aggregate=behavior,
        observer_control_summary=observers,
        free_viewing_reference_feasibility=tmp_path / "missing_feasibility.csv",
        output=output,
    )
    written = _read_csv(output)

    assert list(written[0]) == FIELDNAMES
    assert [row["required_control"] for row in rows] == [
        "DeepGaze IIE free-viewing fixation reference",
        "center-bias free-viewing baseline",
        "SALICON leave-one-observer-out control",
        "modern free-viewing fixation reference",
        "DeepGaze IIE task-search diagnostic reference",
        "center-bias task-search baseline",
        "COCO-Search18 leave-one-observer-out control",
        "task-specific COCO-Search18 baseline",
        "point-fixation metrics separated from map-distribution metrics",
    ]
    assert {row["status"] for row in rows} == {
        "accepted",
        "diagnostic",
        "missing",
        "needs_feasibility_decision",
    }


def test_missing_observer_summary_marks_observer_controls_missing(tmp_path):
    behavior = tmp_path / "behavior.csv"
    _write_behavior_rows(behavior)

    rows = audit_behavioral_controls(
        behavioral_aggregate=behavior,
        observer_control_summary=tmp_path / "missing_observers.csv",
        free_viewing_reference_feasibility=tmp_path / "missing_feasibility.csv",
        output=None,
    )

    observer_rows = [
        row for row in rows if "leave-one-observer-out" in row["required_control"]
    ]
    assert {row["status"] for row in observer_rows} == {"missing"}
    assert all("missing or incomplete" in row["detail"] for row in observer_rows)


def test_audit_keeps_free_viewing_and_task_search_dataset_scopes_separate():
    rows = build_behavioral_control_gap_audit(
        _behavior_rows(),
        _observer_rows(),
        behavioral_artifact="behavior.csv",
        observer_summary_artifact="observers.csv",
    )

    free_rows = [row for row in rows if row["viewing_regime"] == "free_viewing"]
    task_rows = [row for row in rows if row["viewing_regime"] == "task_search"]

    assert free_rows
    assert task_rows
    assert all("COCO-Search18" not in row["dataset_scope"] for row in free_rows)
    assert all("SALICON" not in row["dataset_scope"] for row in task_rows)
    assert all("CAT2000" not in row["dataset_scope"] for row in task_rows)


def test_metric_boundary_row_distinguishes_point_and_map_metrics():
    rows = build_behavioral_control_gap_audit(
        _behavior_rows(),
        _observer_rows(),
        behavioral_artifact="behavior.csv",
        observer_summary_artifact="observers.csv",
    )

    metric_row = next(row for row in rows if row["claim_axis"] == "metric_boundary_control")

    assert metric_row["status"] == "accepted"
    assert "point-fixation metrics" in metric_row["required_control"]
    assert "point-fixation metrics=" in metric_row["detail"]
    assert "nss" in metric_row["detail"]
    assert "auc_judd" in metric_row["detail"]
    assert "map-distribution metrics=" in metric_row["detail"]
    assert "cc" in metric_row["detail"]
    assert "similarity" in metric_row["detail"]
    assert "kl" in metric_row["detail"]
    assert metric_row["viewing_regime"] == "all_regimes_separated"


def test_task_specific_baseline_is_accepted_when_aggregate_rows_exist():
    rows = build_behavioral_control_gap_audit(
        [
            *_behavior_rows(),
            *_task_specific_baseline_rows(),
        ],
        _observer_rows(),
        behavioral_artifact="behavior.csv",
        observer_summary_artifact="observers.csv",
    )

    baseline = next(
        row for row in rows if row["required_control"] == "task-specific COCO-Search18 baseline"
    )

    assert baseline["status"] == "accepted"
    assert baseline["viewing_regime"] == "task_search"
    assert baseline["dataset_scope"] == "COCO-Search18"
    assert "task-conditioned COCO-Search18 prior rows" in baseline["detail"]


def test_feasible_msdb_decision_requires_export_and_preserves_task_diagnostic():
    rows = build_behavioral_control_gap_audit(
        _behavior_rows(),
        _observer_rows(),
        [_msdb_feasibility_row("feasible_now")],
        behavioral_artifact="behavior.csv",
        observer_summary_artifact="observers.csv",
        feasibility_artifact="feasibility.csv",
    )

    modern = next(
        row for row in rows if row["required_control"] == "modern free-viewing fixation reference"
    )
    task_deepgaze = next(
        row for row in rows if row["required_control"] == "DeepGaze IIE task-search diagnostic reference"
    )

    assert modern["status"] == "needs_export_and_evaluation"
    assert modern["current_artifact"] == "feasibility.csv"
    assert "not an accepted behavioral result" in modern["detail"]
    assert task_deepgaze["status"] == "diagnostic"


def test_scored_msdb_rows_accept_modern_free_viewing_reference():
    rows = build_behavioral_control_gap_audit(
        [
            *_behavior_rows(),
            *_msdb_reference_rows(),
        ],
        _observer_rows(),
        [_msdb_feasibility_row("feasible_now")],
        behavioral_artifact="behavior.csv",
        observer_summary_artifact="observers.csv",
        feasibility_artifact="feasibility.csv",
    )

    modern = next(
        row for row in rows if row["required_control"] == "modern free-viewing fixation reference"
    )

    assert modern["status"] == "accepted"
    assert modern["current_artifact"] == "behavior.csv"
    assert "DeepGaze MSDB rows" in modern["detail"]
    assert "SALICON" in modern["detail"]
    assert "CAT2000" in modern["detail"]


def test_deferred_msdb_decision_documents_limitation():
    rows = build_behavioral_control_gap_audit(
        _behavior_rows(),
        _observer_rows(),
        [_msdb_feasibility_row("defer_or_document_limitation")],
        behavioral_artifact="behavior.csv",
        observer_summary_artifact="observers.csv",
        feasibility_artifact="feasibility.csv",
    )

    modern = next(
        row for row in rows if row["required_control"] == "modern free-viewing fixation reference"
    )

    assert modern["status"] == "defer_or_document_limitation"
    assert "accepted DeepGaze-class free-viewing reference" in modern["next_action"]


def _write_behavior_rows(path):
    rows = _behavior_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_observer_rows(path):
    rows = _observer_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _behavior_rows():
    rows = []
    for dataset, protocol in [
        ("salicon_static2000", "points"),
        ("cat2000_static2000", "points"),
        ("coco_search18_static2000", "task_points"),
    ]:
        for model, method, family in [
            ("deepgaze_reference", "deepgaze_precomputed", "reference"),
            ("center_bias_baseline", "center_bias", "baseline"),
        ]:
            for metric in ["nss", "auc_judd", "auc_borji", "shuffled_auc", "cc", "similarity", "kl"]:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "saliency_method": method,
                        "saliency_family": family,
                        "metric": metric,
                        "n": "2",
                        "mean": "0.5",
                        "fixation_protocol": protocol,
                    }
                )
    return rows


def _task_specific_baseline_rows():
    return [
        {
            "dataset": "coco_search18_static2000",
            "model": "coco_search18_task_prior_baseline",
            "saliency_method": "coco_search18_task_prior",
            "saliency_family": "task_search_baseline",
            "metric": metric,
            "n": "2",
            "mean": "0.5",
            "fixation_protocol": "task_points",
        }
        for metric in ["nss", "auc_judd", "auc_borji", "shuffled_auc", "cc", "similarity", "kl"]
    ]


def _msdb_reference_rows():
    rows = []
    for dataset in ["salicon_static2000", "cat2000_static2000"]:
        for metric in ["nss", "auc_judd", "auc_borji", "shuffled_auc", "cc", "similarity", "kl"]:
            rows.append(
                {
                    "dataset": dataset,
                    "model": "deepgaze_msdb_reference",
                    "saliency_method": "deepgaze_precomputed",
                    "saliency_family": "reference",
                    "metric": metric,
                    "n": "2000",
                    "mean": "1.0",
                    "fixation_protocol": "points",
                }
            )
    return rows


def _observer_rows():
    return [
        {
            "dataset": "SALICON",
            "viewing_regime": "free_viewing",
            "control_type": "leave_one_observer_out",
            "status": "complete",
            "row_count": "10",
            "image_count": "2",
            "subject_count": "5",
            "num_observers_min": "4",
            "num_observers_max": "6",
            "mean_inter_observer_nss": "0.5",
            "median_inter_observer_nss": "0.5",
            "mean_inter_observer_auc": "0.9",
            "median_inter_observer_auc": "0.9",
            "source_path": "outputs/observer_controls_v2/salicon.csv",
            "interpretation": "human/interobserver context",
        },
        {
            "dataset": "COCO-Search18",
            "viewing_regime": "task_search",
            "control_type": "leave_one_observer_out",
            "status": "complete",
            "row_count": "8",
            "image_count": "2",
            "subject_count": "4",
            "num_observers_min": "2",
            "num_observers_max": "4",
            "mean_inter_observer_nss": "0.6",
            "median_inter_observer_nss": "0.6",
            "mean_inter_observer_auc": "0.91",
            "median_inter_observer_auc": "0.91",
            "source_path": "outputs/observer_controls_v2/coco.csv",
            "interpretation": "human/interobserver context",
        },
    ]


def _msdb_feasibility_row(decision):
    return {
        "candidate_reference": "DeepGaze MSDB",
        "viewing_regime": "free_viewing",
        "dataset_scope": "SALICON/CAT2000",
        "local_support": "DeepGazeMSDB class available=yes",
        "requires_download": "yes",
        "requires_new_dependency": "no",
        "estimated_run_scope": "4000 free-viewing images",
        "decision": decision,
        "next_action": "test action",
        "detail": "test detail",
    }
