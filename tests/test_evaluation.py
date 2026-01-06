#!/usr/bin/env python3
import subprocess

import pytest

# Process folder paths to get test names from search dir and second-last folder
TEST_CASES = [
    pytest.param(
        "examples/evaluation_continuous.py --rendering 0",
        id="evaluation_continuous",
    ),
    pytest.param(
        "examples/evaluation_multidiscrete.py --blocking 0",
        id="evaluation_multidiscrete",
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
