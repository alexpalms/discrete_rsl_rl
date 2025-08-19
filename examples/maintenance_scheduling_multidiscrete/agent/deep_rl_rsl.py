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
            "num_steps_per_env": 2048,
            "seed": 1,
            "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
            "save_interval": 100,
            "empirical_normalization": None,
            "runner": {
                "checkpoint": -1,
                "experiment_name": "multidiscrete_PPO_rsl",
                "load_run": -1,
                "log_interval": 1,
                "max_iterations": 150,
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
        runner = OnPolicyRunner(env, train_cfg_dict, "./", device="cuda")
        runner.load(self.model_path)
        self.policy = runner.get_inference_policy(device=self.device)

    def get_action(self, obs):
        with torch.no_grad():
            action = self.policy(TensorDict({"policy": obs}))
        return action