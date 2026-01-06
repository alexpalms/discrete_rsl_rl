#!/usr/bin/env python3
import subprocess

import pytest

# Process folder paths to get test names from search dir and second-last folder
TEST_CASES = [
    pytest.param(
        "examples/training_rsl_continuous.py --config ./tests/assets/rsl_config_continuous.yaml",
        id="training_rsl_continuous",
    ),
    pytest.param(
        "examples/training_rsl_multidiscrete.py --config ./tests/assets/rsl_config_multidiscrete.yaml",
        id="training_rsl_multidiscrete",
    ),
    pytest.param(
        "examples/training_sb3_continuous.py --config ./tests/assets/sb3_config_continuous.yaml",
        id="training_sb3_continuous",
    ),
    pytest.param(
        "examples/training_sb3_multidiscrete.py --config ./tests/assets/sb3_config_multidiscrete.yaml",
        id="training_sb3_multidiscrete",
    ),
]


@pytest.mark.parametrize("script", TEST_CASES)
def test_evaluate(script: str):
    command = ["uv", "run", "python"]
    for item in script.split(" "):
        command.append(item)
    result = subprocess.run(  # noqa:S603
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
