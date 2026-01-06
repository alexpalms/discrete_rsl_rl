"""Training script for the continuous PPO agent using Stable Baselines3."""

import argparse
import logging
import os
import re
import sys
from copy import deepcopy

import genesis as gs  # pyright:ignore[reportMissingTypeStubs]
import yaml
from legged_locomotion_continuous.environment.environment import Environment
from sb3.misc import (
    AutoSave,
    CustomMetrics,
    StartingSteps,
    linear_schedule,
    make_sb3_env,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)


def main(config: str):
    with open(config) as file:
        train_config_in = yaml.safe_load(file)

    train_config = deepcopy(train_config_in)
    env_args = {
        "num_envs": train_config["num_envs"],
    }

    local_path = os.path.dirname(os.path.abspath(__file__))

    results_folder = os.path.join(local_path, "../runs", train_config["name"])
    model_folder = os.path.join(results_folder, "model")
    tensor_board_folder = os.path.join(results_folder, "tb")
    monitor_folder = os.path.join(results_folder, "monitor")

    model_save_config = train_config["model_save"]
    training_stop_config = train_config["training_stop"]
    policy_kwargs = train_config["policy_kwargs"]

    os.makedirs(model_folder, exist_ok=True)

    gs.init(logging_level="warning")  # pyright:ignore[reportUnknownMemberType]

    env = make_sb3_env(
        Environment, env_args, seed=train_config["seed"], monitor_folder=monitor_folder
    )
    num_envs = env_args["num_envs"]
    logger.info(f"Activated {num_envs} environment(s)")

    # Generic algo settings
    gamma = train_config["gamma"]
    model_checkpoint_path = train_config["model_checkpoint_path"]
    learning_rate = linear_schedule(
        train_config["learning_rate"][0], train_config["learning_rate"][1]
    )

    clip_range = linear_schedule(
        train_config["clip_range"][0], train_config["clip_range"][1]
    )
    clip_range_vf = clip_range
    n_epochs = train_config["n_epochs"]
    n_steps = train_config["n_steps"]
    batch_size = train_config["batch_size"]
    min_steps = num_envs * n_steps
    assert training_stop_config["max_time_steps"] > min_steps, (  # noqa: S101
        f"The minimum number of training steps is {min_steps}"
    )
    gae_lambda = train_config["gae_lambda"]
    normalize_advantage = train_config["normalize_advantage"]
    ent_coef = train_config["ent_coef"]
    vf_coef = train_config["vf_coef"]
    max_grad_norm = train_config["max_grad_norm"]
    target_kl = train_config["target_kl"]

    starting_steps = 0
    reset_num_timesteps = True
    callbacks: list[BaseCallback] = [CustomMetrics()]

    if model_checkpoint_path is None:
        agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            gamma=gamma,
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_steps=n_steps,
            learning_rate=learning_rate,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            tensorboard_log=tensor_board_folder,
            device="cuda",
        )
    else:
        # Load the trained agent
        # Use regex to find the number after the latest underscore
        match = re.search(r"_(\d+)(?!.*_\d)", model_checkpoint_path)
        if not match:
            raise Exception(
                f"{model_checkpoint_path} should contain a number at the end of the filename indicating the number of training steps."
            )
        starting_steps = int(match.group(1))  # Convert the found number to an integer

        agent = PPO.load(  # pyright:ignore[reportUnknownMemberType]
            model_checkpoint_path,
            env=env,
            policy_kwargs=policy_kwargs,
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_steps=n_steps,
            gamma=gamma,
            learning_rate=learning_rate,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            tensorboard_log=tensor_board_folder,
            device="cuda",
        )
        reset_num_timesteps = False
        callbacks.append(StartingSteps(starting_steps=starting_steps))

    # Print policy network architecture
    logger.info("Policy architecture:")
    logger.info(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave = model_save_config["autosave"]["active"]
    autosave_freq = model_save_config["autosave"]["frequency"]

    if autosave:
        callbacks.append(
            AutoSave(
                check_freq=autosave_freq,
                num_envs=num_envs,
                save_path=model_folder,
                filename_prefix="model_",
                starting_steps=starting_steps,
            )
        )

    # Train the agent
    time_steps = training_stop_config["max_time_steps"]
    agent.learn(  # pyright:ignore[reportUnknownMemberType]
        total_timesteps=time_steps,
        reset_num_timesteps=reset_num_timesteps,
        callback=callbacks,
    )

    # Save the agent
    new_model_checkpoint = "model_" + str(starting_steps + time_steps)
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

    # Free memory
    assert agent.env is not None  # noqa
    agent.env.close()
    del agent.env
    del agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./examples/legged_locomotion_continuous/sb3_config.yaml",
        help="Configuration file",
    )
    opt = parser.parse_args()
    logger.info(opt)
    try:
        main(opt.config)
        sys.exit(0)
    except Exception as exc:
        logger.error(exc)
        sys.exit(1)
