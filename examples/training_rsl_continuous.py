"""Training script for the continuous PPO agent using RSL."""

import argparse
import logging
import os
import sys

import genesis as gs  # pyright:ignore[reportMissingTypeStubs]
import yaml
from legged_locomotion_continuous.environment.environment import Environment

from rsl_rl.runners import OnPolicyRunner  # pyright:ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(config: str) -> None:
    gs.init(logging_level="warning")  # pyright:ignore[reportUnknownMemberType]

    with open(config) as file:
        config_in = yaml.safe_load(file)

    train_config = config_in["train_cfg"]

    local_path = os.path.dirname(os.path.abspath(__file__))

    results_folder = os.path.join(
        local_path, "../runs", train_config["runner"]["experiment_name"]
    )
    log_folder = os.path.join(results_folder, "logs/")

    env = Environment(num_envs=config_in["num_envs"])

    runner = OnPolicyRunner(env, train_config, log_folder, device=gs.device)  # pyright:ignore

    runner.learn(
        num_learning_iterations=train_config["runner"]["max_iterations"],
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    """Run the training script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./examples/legged_locomotion_continuous/rsl_config.yaml",
        help="Configuration file",
    )
    args = parser.parse_args()
    logger.info(args)
    try:
        main(args.config)
        sys.exit(0)
    except Exception as exc:
        logger.error(exc)
        sys.exit(1)
