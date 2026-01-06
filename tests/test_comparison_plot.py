#!/usr/bin/env python3
import subprocess

import pytest

# Process folder paths to get test names from search dir and second-last folder
TEST_CASES = [
    pytest.param(
        "runs/plot.py --blocking 0",
        id="comparison_plot",
    ),
]


@pytest.mark.parametrize("script", TEST_CASES)
def test_comparison_plot(script: str):
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
