import os
import torch
from stable_baselines3 import PPO

class Agent:
    def __init__(self, env):
        model_path = os.path.join(os.path.dirname(__file__), "model_sb3.zip")
        assert os.path.exists(model_path), "Model file not found at {model_path}"

        self.agent = PPO.load(model_path, device="cuda")

    def get_action(self, obs):
        obs = obs.cpu().numpy()
        prediction,_ = self.agent.predict(obs, deterministic=True)
        prediction = torch.tensor(prediction, dtype=torch.long, device="cuda")
        return prediction