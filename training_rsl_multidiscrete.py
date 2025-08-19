import argparse
import os

from rsl_rl.runners import OnPolicyRunner

from examples.maintenance_scheduling_multidiscrete.environment.environment import Environment

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 2048,
        "seed": 1,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "save_interval": 15,
        "empirical_normalization": None,
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
        "policy": {
            "class_name": "ActorCritic",
            "activation": "relu",
            "actor_hidden_dims": [512, 512],
            "critic_hidden_dims": [512, 512],
            "init_noise_std": 1.0,
            "action_type": "multi_discrete",
        },
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.15,
            "desired_kl": None,
            "entropy_coef": 0.001,
            "gamma": 0.95,
            "lam": 0.95,
            "learning_rate": 0.00025,
            "max_grad_norm": 0.5,
            "num_learning_epochs": 4,
            "num_mini_batches": 128,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
    }

    return train_cfg_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multidiscrete_PPO_rsl")
    parser.add_argument("-B", "--num_envs", type=int, default=32)
    parser.add_argument("--max_iterations", type=int, default=150)
    args = parser.parse_args()
    print(args)

    local_path = os.path.dirname(os.path.abspath(__file__))

    results_folder = os.path.join(local_path, "runs", args.exp_name)
    log_folder = os.path.join(results_folder, "logs/")

    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env = Environment(num_envs=args.num_envs)

    runner = OnPolicyRunner(env, train_cfg, log_folder, device="cuda")

    runner.learn(num_learning_iterations=args.max_iterations)


if __name__ == "__main__":
    main()