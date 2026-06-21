"""Run the Paper 1 primary behavioral latent-to-fixation encoding lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hma.behavioral.latent_fixation import (  # noqa: E402
    json_dumps_stable,
    load_fixation_dataset_bundle,
    run_latent_fixation_encoding,
)
from hma.utils.config import load_yaml  # noqa: E402
from hma.utils.paths import resolve_path  # noqa: E402

from paper1_clean_rerun_common import (  # noqa: E402
    PROJECT_ROOT as COMMON_PROJECT_ROOT,
    CleanRerunValidationError,
    audit_row,
    dry_run_report_path,
    load_clean_config,
    model_identity,
    output_paths_from_config,
    pipe_join,
    read_csv,
    require_publication_path,
    run_with_failure_log,
    validate_contract,
    validate_execution_enabled,
    validate_input_path,
    write_csv,
    write_dry_run_report,
    write_lane_audit,
)


LANE = "primary_behavioral_latent_to_fixation_encoding"
OUTPUT_KEYS = [
    "fixation_encoding_scores",
    "fixation_image_scores",
    "feature_reduction_metadata",
    "readout_selection_artifact",
    "audit",
    "legacy_exclusion_audit",
    "failure_log",
]
LEGACY_BEHAVIORAL_PATHS = [
    "outputs/paper1_publication_v0/behavioral/aggregate.csv",
    "outputs/paper1_publication_v0/behavioral/uncertainty.csv",
    "outputs/paper1_publication_v0/behavioral/per_image_metrics",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/paper1_primary_behavioral_latent_fixation.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional execution profile from config.profiles, e.g. local_smoke.",
    )
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help="Skip external artifact hash checks for emergency debugging only.",
    )
    parser.add_argument(
        "--cell-table",
        help="Run one exported behavioral latent cell from a generated cluster table.",
    )
    parser.add_argument(
        "--cell-index",
        type=int,
        help="Row index in --cell-table for one exported behavioral latent cell.",
    )
    parser.add_argument(
        "--cell-output-dir",
        help="Override output directory for one behavioral latent cell.",
    )
    parser.add_argument(
        "--from-cells",
        action="store_true",
        help="Materialize final lane outputs by concatenating scored behavioral latent cells.",
    )
    parser.add_argument(
        "--cell-root",
        default=None,
        help="Root containing scored behavioral latent cells for --from-cells.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_with_failure_log(
        LANE,
        lambda: run(
            args.config,
            dry_run=args.dry_run,
            profile=args.profile,
            max_items=args.max_items,
            verify_hashes=not args.no_verify_hashes,
            cell_table=args.cell_table,
            cell_index=args.cell_index,
            cell_output_dir=args.cell_output_dir,
            from_cells=args.from_cells,
            cell_root=args.cell_root,
        ),
    )


def run(
    config_path: str,
    *,
    dry_run: bool,
    profile: str | None,
    max_items: int | None,
    verify_hashes: bool,
    cell_table: str | None = None,
    cell_index: int | None = None,
    cell_output_dir: str | None = None,
    from_cells: bool = False,
    cell_root: str | None = None,
) -> int:
    config = load_clean_config(config_path, lane=LANE)
    validate_contract(config, require_authorized=not dry_run)
    validate_execution_enabled(config, dry_run=dry_run)
    outputs = output_paths_from_config(config, OUTPUT_KEYS)
    for key in ("fixation_encoding_scores", "fixation_image_scores"):
        if outputs[key].suffix.lower() != ".csv":
            raise CleanRerunValidationError(f"outputs.{key} must be a CSV path")

    profile_name, profile_config = active_profile(config, profile)
    if cell_table is not None or cell_index is not None:
        if dry_run:
            raise CleanRerunValidationError("--dry-run cannot be combined with --cell-table")
        if cell_table is None or cell_index is None:
            raise CleanRerunValidationError("--cell-table and --cell-index must be provided together")
        return run_cell(
            config=config,
            profile_name=profile_name,
            profile_config=profile_config,
            cell_table=cell_table,
            cell_index=cell_index,
            cell_output_dir=cell_output_dir,
            verify_hashes=verify_hashes,
        )

    dataset_rows = selected_dataset_rows(config, profile_config, max_items=max_items)
    model_rows = selected_model_rows(config, profile_config)
    layer_map = selected_layer_map(config, profile_config)
    planned_rows = build_planned_rows(
        dataset_rows=dataset_rows,
        model_rows=model_rows,
        layer_map=layer_map,
        config=config,
        profile_name=profile_name,
        profile_config=profile_config,
    )
    if from_cells:
        if dry_run:
            raise CleanRerunValidationError("--dry-run cannot be combined with --from-cells")
        return collect_cell_outputs(
            config=config,
            outputs=outputs,
            planned_rows=planned_rows,
            profile_name=profile_name,
            cell_root=cell_root,
        )

    legacy_rows = legacy_exclusion_rows()
    write_csv(outputs["legacy_exclusion_audit"], legacy_rows)
    audit_rows = [
        audit_row(
            lane=LANE,
            check_id="publication_paths",
            status="pass",
            artifact="behavioral_latent_fixation_outputs",
            path=pipe_join(str(path.relative_to(COMMON_PROJECT_ROOT)) for path in outputs.values()),
            detail="all configured primary behavioral outputs are under outputs/paper1_publication_v0",
        ),
        audit_row(
            lane=LANE,
            check_id="legacy_behavioral_pipeline_excluded_from_v0",
            status="pass",
            artifact="legacy_behavioral_outputs",
            path=pipe_join(LEGACY_BEHAVIORAL_PATHS),
            detail="old behavioral-map/saliency/scanpath aggregate paths are diagnostic only and excluded from V0 primary evidence",
        ),
        audit_row(
            lane=LANE,
            check_id="planned_latent_fixation_rows",
            status="pass",
            artifact="primary_behavioral_plan",
            path=dry_run_report_path(LANE).relative_to(COMMON_PROJECT_ROOT),
            detail=f"profile={profile_name}; planned_dataset_model_rows={len(planned_rows)}",
        ),
    ]
    audit_path = write_lane_audit(outputs["audit"], audit_rows)
    if dry_run:
        report = write_dry_run_report(
            lane=LANE,
            config_path=config_path,
            expected_outputs=outputs,
            planned_rows=planned_rows,
            audit_path=audit_path,
            extra={
                "profile": profile_name,
                "legacy_behavioral_pipeline_status": "legacy_behavioral_pipeline_excluded_from_v0",
                "datasets": [row["dataset"] for row in dataset_rows],
                "models": [row["model_id"] for row in model_rows],
            },
        )
        print(f"dry_run_passed={report.relative_to(COMMON_PROJECT_ROOT)}")
        return 0

    aggregate_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    reduction_records: list[dict[str, Any]] = []
    encoding_config = dict(config.get("encoding", {}))
    target_size = parse_target_size(encoding_config.get("target_size", [32, 32]))
    for dataset_row in dataset_rows:
        bundle = load_fixation_dataset_bundle(
            dataset_row["dataset"],
            dataset_row["dataset_config"],
            target_size=target_size,
        )
        for model in model_rows:
            model_id = model["model_id"]
            layers = layers_for_model(layer_map, model, profile_config)
            artifact_dir = feature_artifact_path(
                config=config,
                profile_config=profile_config,
                dataset=dataset_row["dataset"],
                model_id=model_id,
            )
            if not artifact_dir.is_dir():
                raise CleanRerunValidationError(
                    "Missing latent feature artifact for primary behavioral run: "
                    f"dataset={dataset_row['dataset']} model={model_id} path={artifact_dir.relative_to(COMMON_PROJECT_ROOT)}"
                )
            score_rows, per_image_rows, selection, reduction = run_latent_fixation_encoding(
                bundle=bundle,
                artifact_dir=artifact_dir,
                model_id=model_id,
                layers=layers,
                ridge_alphas=[float(value) for value in encoding_config.get("ridge_alphas", [1.0])],
                pca_components=int(encoding_config.get("pca_components", 64)),
                train_fraction=float(encoding_config.get("train_fraction", 0.75)),
                validation_fraction_of_train=float(
                    encoding_config.get("validation_fraction_of_train", 0.25)
                ),
                seed=int(config.get("seed", 123)),
                verify_hashes=verify_hashes,
            )
            identity = model_identity(model)
            for row in score_rows:
                row.update(identity)
                row["profile"] = profile_name
            for row in per_image_rows:
                row.update(identity)
                row["profile"] = profile_name
            aggregate_rows.extend(score_rows)
            image_rows.extend(per_image_rows)
            selection_records.append(selection)
            reduction_records.extend(reduction)

    if not aggregate_rows:
        raise CleanRerunValidationError("No latent fixation aggregate rows were produced")
    write_csv(outputs["fixation_encoding_scores"], aggregate_rows)
    write_csv(outputs["fixation_image_scores"], image_rows)
    outputs["feature_reduction_metadata"].write_text(
        json_dumps_stable(
            {
                "schema_version": "hma.paper1.primary_behavioral_feature_reduction.v1",
                "lane": LANE,
                "profile": profile_name,
                "records": reduction_records,
            }
        ),
        encoding="utf-8",
    )
    outputs["readout_selection_artifact"].write_text(
        json_dumps_stable(
            {
                "schema_version": "hma.paper1.primary_behavioral_readout_selection.v1",
                "lane": LANE,
                "profile": profile_name,
                "records": selection_records,
            }
        ),
        encoding="utf-8",
    )
    audit_rows.append(
        audit_row(
            lane=LANE,
            check_id="fixation_encoding_scores_written",
            status="pass",
            artifact="fixation_encoding_scores",
            path=outputs["fixation_encoding_scores"].relative_to(COMMON_PROJECT_ROOT),
            detail=f"profile={profile_name}; rows={len(aggregate_rows)}; image_score_rows={len(image_rows)}",
        )
    )
    write_lane_audit(outputs["audit"], audit_rows)
    print(outputs["fixation_encoding_scores"].relative_to(COMMON_PROJECT_ROOT))
    print(outputs["fixation_image_scores"].relative_to(COMMON_PROJECT_ROOT))
    return 0


def run_cell(
    *,
    config: dict[str, Any],
    profile_name: str,
    profile_config: dict[str, Any],
    cell_table: str,
    cell_index: int,
    cell_output_dir: str | None,
    verify_hashes: bool,
) -> int:
    rows = read_csv(cell_table)
    if cell_index < 0 or cell_index >= len(rows):
        raise CleanRerunValidationError(
            f"Behavioral latent cell index {cell_index} outside table range 0..{len(rows)-1}"
        )
    row = rows[cell_index]
    dataset_name = row.get("dataset", "")
    model_id = row.get("model_id", "")
    if not dataset_name or not model_id:
        raise CleanRerunValidationError("Behavioral latent cell row must include dataset and model_id")

    row_max_items = int(row["max_items"]) if row.get("max_items") else None
    dataset_rows = selected_dataset_rows(config, profile_config, max_items=row_max_items)
    dataset_row = next((item for item in dataset_rows if item["dataset"] == dataset_name), None)
    if dataset_row is None:
        raise CleanRerunValidationError(f"Dataset {dataset_name} is not selected by this config/profile")

    model_rows = selected_model_rows(config, profile_config)
    model = next((item for item in model_rows if item["model_id"] == model_id), None)
    if model is None:
        raise CleanRerunValidationError(f"Model {model_id} is not selected by this config/profile")

    layer_map = selected_layer_map(config, profile_config)
    row_layers = [value for value in str(row.get("layers", "")).split("|") if value]
    layers = row_layers or layers_for_model(layer_map, model, profile_config)
    artifact_dir = require_publication_path(row.get("artifact_dir", ""), "behavior_latent_cell.artifact_dir")
    if not artifact_dir.is_dir():
        raise CleanRerunValidationError(
            f"Missing exported behavioral latent artifact: {artifact_dir.relative_to(COMMON_PROJECT_ROOT)}"
        )
    output_dir = require_publication_path(
        cell_output_dir
        or row.get("output_dir")
        or default_cell_output_dir(config, dataset_name, model_id),
        "behavior_latent_cell.output_dir",
    )

    encoding_config = dict(config.get("encoding", {}))
    bundle = load_fixation_dataset_bundle(
        dataset_name,
        dataset_row["dataset_config"],
        target_size=parse_target_size(encoding_config.get("target_size", [32, 32])),
    )
    score_rows, per_image_rows, selection, reduction = run_latent_fixation_encoding(
        bundle=bundle,
        artifact_dir=artifact_dir,
        model_id=model_id,
        layers=layers,
        ridge_alphas=[float(value) for value in encoding_config.get("ridge_alphas", [1.0])],
        pca_components=int(encoding_config.get("pca_components", 64)),
        train_fraction=float(encoding_config.get("train_fraction", 0.75)),
        validation_fraction_of_train=float(
            encoding_config.get("validation_fraction_of_train", 0.25)
        ),
        seed=int(config.get("seed", 123)),
        verify_hashes=verify_hashes,
    )
    identity = model_identity(model)
    for scored in score_rows:
        scored.update(identity)
        scored["profile"] = profile_name
    for scored in per_image_rows:
        scored.update(identity)
        scored["profile"] = profile_name
    write_cell_outputs(
        output_dir=output_dir,
        dataset=dataset_name,
        model_id=model_id,
        profile_name=profile_name,
        aggregate_rows=score_rows,
        image_rows=per_image_rows,
        selection_records=[selection],
        reduction_records=reduction,
        artifact_dir=artifact_dir,
    )
    print(output_dir.relative_to(COMMON_PROJECT_ROOT))
    return 0


def collect_cell_outputs(
    *,
    config: dict[str, Any],
    outputs: dict[str, Path],
    planned_rows: list[dict[str, Any]],
    profile_name: str,
    cell_root: str | None,
) -> int:
    root = require_publication_path(
        cell_root or default_cell_root(config),
        "behavior_latent_cells.root",
    )
    aggregate_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    reduction_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for planned in planned_rows:
        cell_dir = cell_output_dir(root, planned["dataset"], planned["model_id"])
        aggregate_path = cell_dir / "fixation_encoding_scores.csv"
        image_path = cell_dir / "fixation_image_scores.csv"
        reduction_path = cell_dir / "feature_reduction_metadata.json"
        selection_path = cell_dir / "readout_selection_artifact.json"
        required = [aggregate_path, image_path, reduction_path, selection_path]
        absent = [path.name for path in required if not path.is_file()]
        if absent:
            missing.append(
                f"{planned['dataset']}:{planned['model_id']}:{','.join(absent)}"
            )
            continue
        aggregate_rows.extend(read_csv(aggregate_path))
        image_rows.extend(read_csv(image_path))
        reduction_payload = json.loads(reduction_path.read_text(encoding="utf-8"))
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        reduction_records.extend(list(reduction_payload.get("records", [])))
        selection_records.extend(list(selection_payload.get("records", [])))
    if missing:
        preview = pipe_join(missing[:20])
        suffix = f"; remaining={len(missing) - 20}" if len(missing) > 20 else ""
        raise CleanRerunValidationError(
            "Missing scored behavioral latent cells before materialization: "
            f"{preview}{suffix}"
        )
    if not aggregate_rows:
        raise CleanRerunValidationError("No scored behavioral latent cells were found")

    legacy_rows = legacy_exclusion_rows()
    write_csv(outputs["legacy_exclusion_audit"], legacy_rows)
    write_csv(outputs["fixation_encoding_scores"], aggregate_rows)
    write_csv(outputs["fixation_image_scores"], image_rows)
    outputs["feature_reduction_metadata"].write_text(
        json_dumps_stable(
            {
                "schema_version": "hma.paper1.primary_behavioral_feature_reduction.v1",
                "lane": LANE,
                "profile": profile_name,
                "records": reduction_records,
            }
        ),
        encoding="utf-8",
    )
    outputs["readout_selection_artifact"].write_text(
        json_dumps_stable(
            {
                "schema_version": "hma.paper1.primary_behavioral_readout_selection.v1",
                "lane": LANE,
                "profile": profile_name,
                "records": selection_records,
            }
        ),
        encoding="utf-8",
    )
    audit_rows = [
        audit_row(
            lane=LANE,
            check_id="publication_paths",
            status="pass",
            artifact="behavioral_latent_fixation_outputs",
            path=pipe_join(str(path.relative_to(COMMON_PROJECT_ROOT)) for path in outputs.values()),
            detail="all configured primary behavioral outputs are under outputs/paper1_publication_v0",
        ),
        audit_row(
            lane=LANE,
            check_id="legacy_behavioral_pipeline_excluded_from_v0",
            status="pass",
            artifact="legacy_behavioral_outputs",
            path=pipe_join(LEGACY_BEHAVIORAL_PATHS),
            detail="old behavioral-map/saliency/scanpath aggregate paths are diagnostic only and excluded from V0 primary evidence",
        ),
        audit_row(
            lane=LANE,
            check_id="scored_latent_fixation_cells_materialized",
            status="pass",
            artifact="behavioral_latent_fixation_cells",
            path=str(root.relative_to(COMMON_PROJECT_ROOT)),
            detail=(
                f"profile={profile_name}; cells={len(planned_rows)}; "
                f"aggregate_rows={len(aggregate_rows)}; image_score_rows={len(image_rows)}"
            ),
        ),
    ]
    write_lane_audit(outputs["audit"], audit_rows)
    print(outputs["fixation_encoding_scores"].relative_to(COMMON_PROJECT_ROOT))
    print(outputs["fixation_image_scores"].relative_to(COMMON_PROJECT_ROOT))
    return 0


def write_cell_outputs(
    *,
    output_dir: Path,
    dataset: str,
    model_id: str,
    profile_name: str,
    aggregate_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    selection_records: list[dict[str, Any]],
    reduction_records: list[dict[str, Any]],
    artifact_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fixation_encoding_scores.csv", aggregate_rows)
    write_csv(output_dir / "fixation_image_scores.csv", image_rows)
    (output_dir / "feature_reduction_metadata.json").write_text(
        json_dumps_stable(
            {
                "schema_version": "hma.paper1.primary_behavioral_feature_reduction.v1",
                "lane": LANE,
                "profile": profile_name,
                "records": reduction_records,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "readout_selection_artifact.json").write_text(
        json_dumps_stable(
            {
                "schema_version": "hma.paper1.primary_behavioral_readout_selection.v1",
                "lane": LANE,
                "profile": profile_name,
                "records": selection_records,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json_dumps_stable(
            {
                "schema_version": "hma.paper1.primary_behavioral_latent_cell.v1",
                "lane": LANE,
                "profile": profile_name,
                "dataset": dataset,
                "model_id": model_id,
                "artifact_dir": artifact_dir.relative_to(COMMON_PROJECT_ROOT),
            }
        ),
        encoding="utf-8",
    )


def active_profile(config: dict[str, Any], requested: str | None) -> tuple[str, dict[str, Any]]:
    if requested is None:
        requested = str(config.get("default_profile", "publication_full"))
    profiles = dict(config.get("profiles", {}))
    profile_config = dict(profiles.get(requested, {}))
    if requested != "publication_full" and not profile_config:
        raise CleanRerunValidationError(f"Unknown latent fixation profile: {requested}")
    return requested, profile_config


def selected_dataset_rows(
    config: dict[str, Any],
    profile_config: dict[str, Any],
    *,
    max_items: int | None,
) -> list[dict[str, Any]]:
    datasets = dict(config.get("datasets", {}))
    selected = profile_config.get("datasets")
    names = [str(value) for value in selected] if selected else [
        name for name, section in datasets.items() if section.get("enabled", True) is not False
    ]
    rows = []
    for name in names:
        if name not in datasets:
            raise CleanRerunValidationError(f"Unknown latent fixation dataset: {name}")
        section = dict(datasets[name])
        manifest = validate_input_path(
            section.get("manifest_path") or section.get("manifest"),
            f"datasets.{name}.manifest",
        )
        root = resolve_path(section.get("root", "."))
        if not root.is_dir():
            raise CleanRerunValidationError(f"Dataset root does not exist for {name}: {root}")
        if max_items is not None:
            section["max_items"] = int(max_items)
        rows.append(
            {
                "dataset": name,
                "regime": str(section.get("regime", "")),
                "split": str(section.get("split", "")),
                "manifest": str(manifest.relative_to(COMMON_PROJECT_ROOT)),
                "dataset_config": section,
            }
        )
    if not rows:
        raise CleanRerunValidationError("No datasets selected for primary behavioral lane")
    return rows


def selected_model_rows(
    config: dict[str, Any],
    profile_config: dict[str, Any],
) -> list[dict[str, str]]:
    eligibility = {
        row["model_id"]: row
        for row in read_csv(config.get("model_eligibility_table"))
        if row.get("model_id")
    }
    certification = {
        row["model_id"]: row
        for row in read_csv(config.get("model_certification_summary"))
        if row.get("model_id")
    }
    requested = profile_config.get("models") or dict(config.get("models", {})).get("include")
    requested_ids = [str(value) for value in requested] if requested else sorted(eligibility)
    rows = []
    for model_id in requested_ids:
        row = {**certification.get(model_id, {}), **eligibility.get(model_id, {})}
        if not row:
            raise CleanRerunValidationError(f"Unknown model in primary behavioral lane: {model_id}")
        if row.get("certification_status") != "adapter_certified":
            continue
        if row.get("eligible_latent_neural_encoding") != "yes":
            continue
        if row.get("paper_evidence_status") == "diagnostic_only":
            continue
        rows.append(row)
    if not rows:
        raise CleanRerunValidationError("No certified latent-ready models selected")
    return rows


def selected_layer_map(
    config: dict[str, Any],
    profile_config: dict[str, Any],
) -> dict[str, list[str]]:
    layer_map: dict[str, list[str]] = {}
    source = dict(config.get("models", {})).get("layers_from")
    if source:
        neural_config = load_yaml(resolve_path(source))
        for section_name in ("image_only", "conditioned"):
            for model_id, value in dict(neural_config.get("models", {}).get(section_name, {})).items():
                if isinstance(value, dict):
                    layer_values = value.get("layers", [])
                else:
                    layer_values = value
                layer_map[str(model_id)] = [str(layer) for layer in layer_values]
    for model_id, layers in dict(config.get("model_layers", {})).items():
        layer_map[str(model_id)] = [str(layer) for layer in layers]
    for model_id, layers in dict(profile_config.get("layers", {})).items():
        layer_map[str(model_id)] = [str(layer) for layer in layers]
    if not layer_map:
        raise CleanRerunValidationError("No latent layer map configured")
    return layer_map


def layers_for_model(
    layer_map: dict[str, list[str]],
    model: dict[str, str],
    profile_config: dict[str, Any],
) -> list[str]:
    model_id = model["model_id"]
    layer_limit = profile_config.get("layer_limit")
    layers = list(layer_map.get(model_id) or [
        value for value in str(model.get("latent_tensor_layers", "")).split("|") if value
    ])
    if not layers:
        raise CleanRerunValidationError(f"No latent layers configured for {model_id}")
    if layer_limit is not None:
        layers = layers[: int(layer_limit)]
    return layers


def feature_artifact_path(
    *,
    config: dict[str, Any],
    profile_config: dict[str, Any],
    dataset: str,
    model_id: str,
) -> Path:
    artifacts = dict(config.get("feature_artifacts", {}))
    profile_artifacts = dict(profile_config.get("feature_artifacts", {}))
    explicit = profile_artifacts.get(model_id) or profile_artifacts.get(f"{dataset}:{model_id}")
    if explicit:
        return require_publication_path(explicit, f"profile.feature_artifacts.{dataset}.{model_id}")
    root = require_publication_path(
        artifacts.get("root", "outputs/paper1_publication_v0/behavioral_latent_fixation/external"),
        "feature_artifacts.root",
    )
    return root / dataset / safe_name(model_id)


def build_planned_rows(
    *,
    dataset_rows: list[dict[str, Any]],
    model_rows: list[dict[str, str]],
    layer_map: dict[str, list[str]],
    config: dict[str, Any],
    profile_name: str,
    profile_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in dataset_rows:
        for model in model_rows:
            layers = layers_for_model(layer_map, model, profile_config)
            artifact_dir = feature_artifact_path(
                config=config,
                profile_config=profile_config,
                dataset=dataset["dataset"],
                model_id=model["model_id"],
            )
            rows.append(
                {
                    **model_identity(model),
                    "profile": profile_name,
                    "dataset": dataset["dataset"],
                    "behavioral_regime": dataset["regime"],
                    "behavioral_object": str(
                        dataset["dataset_config"].get(
                            "behavioral_object",
                            "latent_encoded_human_fixation_density",
                        )
                    ),
                    "split": dataset["split"],
                    "manifest": dataset["manifest"],
                    "image_root": str(
                        resolve_path(dataset["dataset_config"].get("root", ".")).relative_to(
                            COMMON_PROJECT_ROOT
                        )
                    ),
                    "artifact_key": str(dataset["dataset_config"].get("artifact_key", "map_key")),
                    "max_items": str(dataset["dataset_config"].get("max_items", "")),
                    "layers": pipe_join(layers),
                    "artifact_dir": str(artifact_dir.relative_to(COMMON_PROJECT_ROOT)),
                    "artifact_exists_now": "yes" if artifact_dir.is_dir() else "no",
                    "evidence_status": "planned_primary_behavioral_latent_to_fixation_encoding",
                    "legacy_behavioral_pipeline_status": "legacy_behavioral_pipeline_excluded_from_v0",
                }
            )
    return rows


def legacy_exclusion_rows() -> list[dict[str, str]]:
    rows = []
    for path in LEGACY_BEHAVIORAL_PATHS:
        resolved = resolve_path(path)
        rows.append(
            {
                "artifact": "legacy_behavioral_pipeline",
                "path": path,
                "exists_now": "yes" if resolved.exists() else "no",
                "v0_status": "legacy_behavioral_pipeline_excluded_from_v0",
                "allowed_use": "diagnostic_expected_range_checks_only",
                "primary_behavioral_evidence": "no",
            }
        )
    return rows


def parse_target_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CleanRerunValidationError("encoding.target_size must be an int or length-2 list")
    return (int(value[0]), int(value[1]))


def default_cell_root(config: dict[str, Any]) -> Path:
    root = require_publication_path(
        config.get("output_root", "outputs/paper1_publication_v0/behavioral_latent_fixation"),
        "output_root",
    )
    return root / "cells"


def default_cell_output_dir(config: dict[str, Any], dataset: str, model_id: str) -> Path:
    return cell_output_dir(default_cell_root(config), dataset, model_id)


def cell_output_dir(root: Path, dataset: str, model_id: str) -> Path:
    return root / safe_name(dataset) / safe_name(model_id)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
