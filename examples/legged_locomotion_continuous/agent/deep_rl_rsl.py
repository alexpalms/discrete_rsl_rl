import os
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

class Agent:
    def __init__(self, env, model_path="./model_rsl.pt", device="cuda", deterministic=True):
        self.env = env
        self.deterministic = deterministic
        self.device = device
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Cannot create agent, policy file '{self.model_path}' not found!")
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
                "experiment_name": "continuous_PPO_rsl",
                "load_run": -1,
                "log_interval": 1,
                "max_iterations": 101,
                "record_interval": -1,
                "resume": False,
                "resume_path": None,
                "run_name": "",
            },
        }

        runner = OnPolicyRunner(env, train_cfg_dict, "./", device="cuda")
        runner.load(self.model_path)
        self.policy = runner.get_inference_policy(device=self.device)

    def get_action(self, obs):
        with torch.no_grad():
            action = self.policy(TensorDict({"policy": obs}))
        return action