import argparse
import os
from examples.legged_locomotion_continuous.environment.environment import get_cfgs

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from examples.legged_locomotion_continuous.environment.environment import Environment


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "seed": 1,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "save_interval": 100,
        "empirical_normalization": None,
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
    }

    return train_cfg_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="continuous_PPO_rsl")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    local_path = os.path.dirname(os.path.abspath(__file__))

    results_folder = os.path.join(local_path, "runs", args.exp_name)
    log_folder = os.path.join(results_folder, "logs/")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)


    env = Environment(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_folder, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
