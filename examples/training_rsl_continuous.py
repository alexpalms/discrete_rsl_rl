"""Training script for the continuous PPO agent using RSL."""

import argparse
import logging
import os

import genesis as gs  # type: ignore
import yaml

from discrete_rsl_rl.runners import OnPolicyRunner
from examples.legged_locomotion_continuous.environment.environment import Environment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
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

    gs.init(logging_level="warning")  # pyright: ignore[reportUnknownMemberType]

    with open(args.config) as file:
        config = yaml.safe_load(file)

    train_config = config["train_cfg"]

    local_path = os.path.dirname(os.path.abspath(__file__))

    results_folder = os.path.join(
        local_path, "runs", train_config["runner"]["experiment_name"]
    )
    log_folder = os.path.join(results_folder, "logs/")

    env = Environment(num_envs=config["num_envs"])

    runner = OnPolicyRunner(env, train_config, log_folder, device=gs.device)  # pyright: ignore

    runner.learn(
        num_learning_iterations=train_config["runner"]["max_iterations"],
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
