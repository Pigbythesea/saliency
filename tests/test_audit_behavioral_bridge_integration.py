from scripts.audit_behavioral_bridge_integration import (
    build_behavioral_bridge_integration_audit,
)


def _base_row(
    *,
    dataset="salicon_static2000",
    model="resnet50",
    method="gradcam",
    family="class_localization",
    metric="nss",
):
    return {
        "dataset": dataset,
        "model": model,
        "saliency_method": method,
        "saliency_family": family,
        "metric": metric,
        "mean": "0.1",
    }


def _relevance_row(
    *,
    dataset="salicon_static2000",
    model="vit_small_patch14_dinov2",
    metric="nss",
):
    return _base_row(
        dataset=dataset,
        model=model,
        method="transformer_relevance",
        family="transformer_relevance",
        metric=metric,
    )


def _by_check(rows, check):
    return next(row for row in rows if row["check"] == check)


def test_behavioral_bridge_integration_audit_passes_expected_rows():
    base_rows = [
        _base_row(),
        _base_row(
            model="vit_small_patch14_dinov2",
            method="attention_rollout",
            family="internal_routing",
        ),
    ]
    relevance_rows = [
        _relevance_row(dataset="salicon_static2000", metric="nss"),
        _relevance_row(dataset="cat2000_static2000", metric="cc"),
    ]
    rows = build_behavioral_bridge_integration_audit(
        base_rows,
        relevance_rows,
        [*base_rows, *relevance_rows],
        base_artifact="base.csv",
        transformer_relevance_artifact="relevance.csv",
        merged_artifact="merged.csv",
        expected_base_rows=2,
        expected_transformer_relevance_rows=2,
        expected_merged_rows=4,
    )

    assert {row["status"] for row in rows} == {"pass"}


def test_behavioral_bridge_integration_audit_fails_dropped_relevance_row():
    base_rows = [
        _base_row(
            model="vit_small_patch14_dinov2",
            method="attention_rollout",
            family="internal_routing",
        )
    ]
    relevance_rows = [
        _relevance_row(dataset="salicon_static2000", metric="nss"),
        _relevance_row(dataset="cat2000_static2000", metric="cc"),
    ]
    rows = build_behavioral_bridge_integration_audit(
        base_rows,
        relevance_rows,
        [base_rows[0], relevance_rows[0]],
        base_artifact="base.csv",
        transformer_relevance_artifact="relevance.csv",
        merged_artifact="merged.csv",
        expected_base_rows=1,
        expected_transformer_relevance_rows=2,
        expected_merged_rows=3,
    )

    assert _by_check(rows, "transformer_relevance_rows_retained")["status"] == "fail"
    assert _by_check(rows, "merged_bridge_row_count")["status"] == "fail"


def test_behavioral_bridge_integration_audit_fails_wrong_relevance_dataset():
    base_rows = [
        _base_row(
            model="vit_small_patch14_dinov2",
            method="attention_rollout",
            family="internal_routing",
        )
    ]
    relevance_rows = [
        _relevance_row(dataset="salicon_static2000", metric="nss"),
        _relevance_row(dataset="coco_search18_static2000", metric="cc"),
    ]
    rows = build_behavioral_bridge_integration_audit(
        base_rows,
        relevance_rows,
        [*base_rows, *relevance_rows],
        base_artifact="base.csv",
        transformer_relevance_artifact="relevance.csv",
        merged_artifact="merged.csv",
        expected_base_rows=1,
        expected_transformer_relevance_rows=2,
        expected_merged_rows=3,
    )

    assert (
        _by_check(rows, "merged_transformer_relevance_dataset_scope")["status"]
        == "fail"
    )


def test_behavioral_bridge_integration_audit_fails_rollout_family_collapse():
    base_rows = [
        _base_row(
            model="vit_small_patch14_dinov2",
            method="attention_rollout",
            family="internal_routing",
        )
    ]
    relevance_rows = [
        _relevance_row(dataset="salicon_static2000", metric="nss"),
        _relevance_row(dataset="cat2000_static2000", metric="cc"),
    ]
    merged_rows = [
        _base_row(
            model="vit_small_patch14_dinov2",
            method="attention_rollout",
            family="transformer_relevance",
        ),
        *relevance_rows,
    ]
    rows = build_behavioral_bridge_integration_audit(
        base_rows,
        relevance_rows,
        merged_rows,
        base_artifact="base.csv",
        transformer_relevance_artifact="relevance.csv",
        merged_artifact="merged.csv",
        expected_base_rows=1,
        expected_transformer_relevance_rows=2,
        expected_merged_rows=3,
    )

    assert _by_check(rows, "attention_rollout_family_preserved")["status"] == "fail"
