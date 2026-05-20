import csv

from scripts.merge_behavioral_aggregates import merge_behavioral_aggregates


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_merge_behavioral_aggregates_appends_ssl_rows_and_preserves_columns(tmp_path):
    fieldnames = [
        "dataset",
        "model",
        "saliency_method",
        "saliency_family",
        "metric",
        "mean",
        "n",
    ]
    base = tmp_path / "base.csv"
    ssl = tmp_path / "ssl.csv"
    _write_csv(
        base,
        [
            {
                "dataset": "salicon_static2000",
                "model": "resnet50",
                "saliency_method": "gradcam",
                "saliency_family": "class_localization",
                "metric": "nss",
                "mean": "0.34",
                "n": "2000",
            }
        ],
        fieldnames,
    )
    _write_csv(
        ssl,
        [
            {
                "dataset": "salicon_static2000",
                "model": "vit_small_patch14_dinov2",
                "saliency_method": "attention_rollout",
                "saliency_family": "internal_routing",
                "metric": "nss",
                "mean": "0.41",
                "n": "2000",
            }
        ],
        fieldnames,
    )

    output = merge_behavioral_aggregates([base, ssl], tmp_path / "merged.csv")

    rows = _read_csv(output)
    assert list(rows[0]) == fieldnames
    assert [row["model"] for row in rows] == [
        "resnet50",
        "vit_small_patch14_dinov2",
    ]


def test_merge_behavioral_aggregates_replaces_exact_duplicate_keys_with_later_rows(tmp_path):
    fieldnames = [
        "dataset",
        "model",
        "saliency_method",
        "saliency_family",
        "metric",
        "mean",
    ]
    base = tmp_path / "base.csv"
    ssl = tmp_path / "ssl.csv"
    duplicate_key = {
        "dataset": "salicon_static2000",
        "model": "vit_small_patch14_dinov2",
        "saliency_method": "attention_rollout",
        "saliency_family": "internal_routing",
        "metric": "nss",
    }
    _write_csv(base, [{**duplicate_key, "mean": "0.1"}], fieldnames)
    _write_csv(ssl, [{**duplicate_key, "mean": "0.2"}], fieldnames)

    output = merge_behavioral_aggregates([base, ssl], tmp_path / "merged.csv")

    rows = _read_csv(output)
    assert len(rows) == 1
    assert rows[0]["mean"] == "0.2"
