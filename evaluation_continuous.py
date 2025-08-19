from examples.legged_locomotion_continuous.environment.environment import Environment
from examples.legged_locomotion_continuous.agent.deep_rl_sb3 import Agent as DeepRlAgentSB3
from examples.legged_locomotion_continuous.agent.deep_rl_rsl import Agent as DeepRlAgentRSL
import matplotlib.pyplot as plt
import torch
from examples.legged_locomotion_continuous.environment.environment import get_cfgs
import genesis as gs

def agent_run(policy, n_episodes):
    num_envs = 1
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env = Environment(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True)
    if policy == "deep_rl_sb3":
        agent = DeepRlAgentSB3(env)
    elif policy == "deep_rl_rsl":
        agent = DeepRlAgentRSL(env)
    else:
        raise NotImplementedError("Policy not implemented")
    obs, info = env.reset()
    risk_history = []
    cumulative_risk = 0.0
    i_episode = 0
    while i_episode < n_episodes:
        action = agent.get_action(obs["policy"])
        obs, reward, done, info = env.step(action)
        cumulative_risk -= reward[0].cpu().numpy()
        risk_history.append(cumulative_risk)
        if torch.any(done):
            print("==========================")
            print(f"Final Risk for policy \"{policy}\": {cumulative_risk}")
            obs, info = env.reset()
            i_episode += 1
            cumulative_risk = 0.0

    return risk_history

if __name__ == "__main__":
    #policies = ["deep_rl_sb3", "deep_rl_rsl"]
    policies = ["deep_rl_rsl"]
    n_episodes = 1
    gs.init(logging_level="warning")

    risk_chart = {}

    for policy in policies:
        risk_chart[policy] = agent_run(policy, n_episodes)

    plt.figure(figsize=(10,6))
    font = {'size': 14}
    plt.rc('font', **font)
    plt.plot(risk_chart["deep_rl_sb3"], label="Deep RL SB3", color='orange', marker='+')
    plt.plot(risk_chart["deep_rl_rsl"], label="Deep RL RSL", color='purple', marker='x')
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Risk', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.title('Risk Over Time', fontsize=16)
    plt.show(block=True)
    pass