"""Training script for the multi-discrete PPO agent using RSL."""

import argparse
import logging
import os

import yaml

from examples.maintenance_scheduling_multidiscrete.environment.environment import (
    Environment,
)
from rsl_rl.runners import OnPolicyRunner  # pyright:ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    """Run the training script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./examples/maintenance_scheduling_multidiscrete/rsl_config.yaml",
        help="Configuration file",
    )
    args = parser.parse_args()
    logger.info(args)

    local_path = os.path.dirname(os.path.abspath(__file__))

    with open(args.config) as file:
        config = yaml.safe_load(file)

    train_config = config["train_cfg"]

    results_folder = os.path.join(
        local_path, "runs", train_config["runner"]["experiment_name"]
    )
    log_folder = os.path.join(results_folder, "logs/")

    env = Environment(num_envs=config["num_envs"])

    runner = OnPolicyRunner(env, train_config, log_folder, device="cuda")

    runner.learn(num_learning_iterations=train_config["runner"]["max_iterations"])


if __name__ == "__main__":
    main()
