import os
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
import yaml

class Agent:
    def __init__(self, env, model_path="./model_rsl.pt", config_path="../rsl_config.yaml", device="cuda", deterministic=True):
        self.env = env
        self.deterministic = deterministic
        self.device = device
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Cannot create agent, policy file '{self.model_path}' not found!")

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path), 'r') as file:
            config = yaml.safe_load(file)

        train_config = config["train_cfg"]

        runner = OnPolicyRunner(env, train_config, "./", device="cuda")
        runner.load(self.model_path)
        self.policy = runner.get_inference_policy(device=self.device)

    def get_action(self, obs):
        with torch.no_grad():
            action = self.policy(TensorDict({"policy": obs}))
        return action