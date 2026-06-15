import pytest

from hma.analysis.cross_axis import (
    cross_axis_panel_preflight,
    family_block_bootstrap,
    leave_one_family_sensitivity,
)


def _rows():
    return [
        {
            "model_id": f"m{index}",
            "family": family,
            "role": role,
            "behavioral_regime": "free_viewing",
            "behavioral_object": "point_fixation_map",
            "behavior": float(index),
            "neural": float(index) + offset,
        }
        for index, (family, role, offset) in enumerate(
            [
                ("cnn", "representation", 0.1),
                ("cnn", "representation", 0.2),
                ("vit", "representation", 0.1),
                ("vit", "efficient", 0.2),
                ("gaze", "gaze", 0.3),
                ("gaze", "gaze", 0.1),
            ]
        )
    ]


def test_cross_axis_preflight_and_leave_one_family():
    preflight = cross_axis_panel_preflight(_rows())
    sensitivity = leave_one_family_sensitivity(
        _rows(), x_key="behavior", y_key="neural"
    )

    assert preflight["passed"] is True
    assert len(sensitivity) == 3
    assert {row["omitted_family"] for row in sensitivity} == {
        "cnn",
        "gaze",
        "vit",
    }


def test_family_block_bootstrap_is_reproducible():
    first = family_block_bootstrap(
        _rows(),
        x_key="behavior",
        y_key="neural",
        resamples=100,
        seed=3,
    )
    second = family_block_bootstrap(
        _rows(),
        x_key="behavior",
        y_key="neural",
        resamples=100,
        seed=3,
    )

    assert first == second
    assert first.uncertainty_unit == "model_family"
    assert first.ci_low <= first.estimate <= first.ci_high


def test_cross_axis_preflight_rejects_regime_mixing():
    rows = _rows()
    rows[0]["behavioral_regime"] = "task_search"

    with pytest.raises(ValueError, match="may not pool behavioral regimes"):
        cross_axis_panel_preflight(rows)
