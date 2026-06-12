from scripts.create_attribution_family_interpretation import (
    audit_attribution_family_interpretation,
    build_attribution_family_interpretation,
)


def _behavior_row(
    *,
    dataset="salicon_static2000",
    family="transformer_relevance",
    method="transformer_relevance",
    model="vit_small_patch14_dinov2",
    metric="nss",
    mean="1.0",
    n="2000",
):
    return {
        "dataset": dataset,
        "model": model,
        "saliency_method": method,
        "saliency_family": family,
        "metric": metric,
        "n": n,
        "mean": mean,
        "ci95_low": str(float(mean) - 0.1),
        "ci95_high": str(float(mean) + 0.1),
    }


def _cross_row(
    *,
    dataset="salicon_static2000",
    family="transformer_relevance",
    method="transformer_relevance",
    metric="nss",
    status="complete",
    n_models="4",
    spearman_encoding="0.5",
    spearman_geometry="0.8",
):
    return {
        "behavior_dataset": dataset,
        "behavior_metric": metric,
        "behavior_saliency_method": method,
        "behavior_saliency_family": family,
        "status": status,
        "n_models": n_models,
        "spearman_behavior_vs_noise_normalized": spearman_encoding,
        "spearman_behavior_vs_geometry": spearman_geometry,
    }


def _overlap_row(
    *,
    dataset="salicon_static2000",
    family="transformer_relevance",
    method="transformer_relevance",
    metric="nss",
    encoding_match="true",
    rsa_match="false",
):
    return {
        "behavior_dataset": dataset,
        "behavior_metric": metric,
        "behavior_saliency_method": method,
        "behavior_saliency_family": family,
        "matches_encoding_leader": encoding_match,
        "matches_rsa_leader": rsa_match,
    }


def _comparison_row(*, dataset="salicon_static2000", metric="nss", delta="0.42"):
    return {
        "dataset": dataset,
        "metric": metric,
        "comparison": "best_transformer_relevance_vs_best_internal_routing",
        "delta_positive_favors_left": delta,
    }


def _by_family(rows, dataset, family):
    return next(
        row
        for row in rows
        if row["dataset"] == dataset and row["saliency_family"] == family
    )


def test_interpretation_table_keeps_transformer_relevance_separate_from_rollout():
    rows = build_attribution_family_interpretation(
        behavioral_rows=[
            _behavior_row(metric="nss", mean="1.2"),
            _behavior_row(metric="cc", mean="0.4"),
            _behavior_row(
                family="internal_routing",
                method="attention_rollout",
                metric="nss",
                mean="0.7",
            ),
        ],
        key_comparison_rows=[
            _comparison_row(metric="nss", delta="0.5"),
            _comparison_row(metric="cc", delta="0.2"),
        ],
        family_ranking_rows=[
            {
                "dataset": "salicon_static2000",
                "saliency_family": "transformer_relevance",
                "metric": "nss",
                "total_n": "8000",
            }
        ],
        cross_level_rows=[
            _cross_row(metric="nss"),
            _cross_row(metric="cc", spearman_encoding="0.7", spearman_geometry="0.9"),
            _cross_row(
                family="internal_routing",
                method="attention_rollout",
                metric="nss",
            ),
        ],
        leader_overlap_rows=[
            _overlap_row(metric="nss"),
            _overlap_row(metric="cc", encoding_match="false", rsa_match="true"),
            _overlap_row(
                family="internal_routing",
                method="attention_rollout",
                metric="nss",
            ),
        ],
    )

    relevance = _by_family(rows, "salicon_static2000", "transformer_relevance")
    rollout = _by_family(rows, "salicon_static2000", "internal_routing")

    assert relevance["included_methods"] == "transformer_relevance"
    assert rollout["included_methods"] == "attention_rollout"
    assert relevance["behavior_metric_rows"] == "2"
    assert relevance["behavior_total_n"] == "8000"
    assert relevance["cross_level_complete_rows"] == "2"
    assert relevance["leader_overlap_rows"] == "2"
    assert relevance["transformer_relevance_vs_rollout_rows"] == "2"
    assert relevance["transformer_relevance_vs_rollout_all_metrics_better"] == "true"
    assert relevance["transformer_relevance_vs_rollout_nss_delta"] == "0.5"
    assert (
        relevance["paper_interpretation"]
        == "transformer_relevance_improves_rollout_behavior_only"
    )
    assert rollout["paper_interpretation"] == "attribution_family_descriptive_only"


def test_interpretation_table_labels_sparse_cross_axis_groups_insufficient():
    rows = build_attribution_family_interpretation(
        behavioral_rows=[
            _behavior_row(
                dataset="cat2000_static2000",
                family="evidence_sensitivity",
                method="vanilla_gradient",
            )
        ],
        cross_level_rows=[
            _cross_row(
                dataset="cat2000_static2000",
                family="evidence_sensitivity",
                method="vanilla_gradient",
                status="insufficient_models",
                n_models="2",
            )
        ],
    )

    row = _by_family(rows, "cat2000_static2000", "evidence_sensitivity")
    assert row["cross_level_insufficient_rows"] == "1"
    assert row["cross_level_n_models_max"] == "2"
    assert row["paper_interpretation"] == "insufficient_cross_axis_evidence"


def test_interpretation_table_labels_reference_and_task_controls():
    rows = build_attribution_family_interpretation(
        behavioral_rows=[
            _behavior_row(
                family="reference",
                method="deepgaze_precomputed",
                model="deepgaze_msdb_reference",
            ),
            _behavior_row(
                dataset="coco_search18_static2000",
                family="task_search_baseline",
                method="coco_search18_task_prior",
                model="coco_search18_task_prior_baseline",
            ),
        ]
    )

    reference = _by_family(rows, "salicon_static2000", "reference")
    task = _by_family(rows, "coco_search18_static2000", "task_search_baseline")
    assert reference["paper_interpretation"] == "reference_control_family"
    assert task["paper_interpretation"] == "task_search_control_family"


def test_interpretation_audit_catches_missing_counts_and_coco_relevance():
    rows = [
        {
            "dataset": "coco_search18_static2000",
            "saliency_family": "transformer_relevance",
            "behavior_metric_rows": "",
            "cross_level_rows": "1",
            "leader_overlap_rows": "1",
            "paper_interpretation": "operational attention",
        },
        {
            "dataset": "salicon_static2000",
            "saliency_family": "internal_routing",
            "behavior_metric_rows": "1",
            "cross_level_rows": "1",
            "leader_overlap_rows": "1",
            "paper_interpretation": "attribution_family_descriptive_only",
        },
    ]

    audit = audit_attribution_family_interpretation(rows)
    statuses = {row["check"]: row["status"] for row in audit}
    assert statuses["transformer_relevance_present"] == "pass"
    assert statuses["rollout_family_separate"] == "pass"
    assert statuses["family_counts_present"] == "fail"
    assert statuses["no_coco_transformer_relevance"] == "fail"
    assert statuses["transformer_interpretation_claim_hygiene"] == "fail"
