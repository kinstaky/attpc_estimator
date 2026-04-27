from __future__ import annotations

from attpc_estimator.cli.config import table_config_values


def test_table_config_values_reads_nested_dotted_tables() -> None:
    payload = {
        "findline": {
            "ransac": {
                "residual_threshold": 20.0,
                "max_trials": 50,
                "ignored": 123,
            },
            "mergeline": {
                "distance_threshold": 15,
                "angle_threshold": 3,
            },
        }
    }

    ransac = table_config_values(
        payload,
        table="findline.ransac",
        allowed_keys={"residual_threshold", "max_trials"},
    )
    mergeline = table_config_values(
        payload,
        table="findline.mergeline",
        allowed_keys={"distance_threshold", "angle_threshold"},
    )

    assert ransac == {
        "residual_threshold": 20.0,
        "max_trials": 50,
    }
    assert mergeline == {
        "distance_threshold": 15,
        "angle_threshold": 3,
    }


def test_table_config_values_returns_empty_dict_for_missing_nested_table() -> None:
    payload = {"findline": {}}

    assert table_config_values(
        payload,
        table="findline.ransac",
        allowed_keys={"residual_threshold"},
    ) == {}
