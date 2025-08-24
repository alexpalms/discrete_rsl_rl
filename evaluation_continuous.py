from examples.legged_locomotion_continuous.environment.environment import Environment
from examples.legged_locomotion_continuous.agent.deep_rl_sb3 import Agent as DeepRlAgentSB3
from examples.legged_locomotion_continuous.agent.deep_rl_rsl import Agent as DeepRlAgentRSL
import genesis as gs

def agent_run(policy, n_episodes):
    num_envs = 1
    env = Environment(num_envs, show_viewer=True)
    if policy == "deep_rl_sb3":
        agent = DeepRlAgentSB3(env)
    elif policy == "deep_rl_rsl":
        agent = DeepRlAgentRSL(env)
    else:
        raise NotImplementedError("Policy not implemented")
    obs, _ = env.reset()
    i_episode = 0
    total_reward = 0.0
    while i_episode < n_episodes:
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        if bool(done[0]):
            print(f"Total reward for policy \"{policy}\": {total_reward}")
            obs, _ = env.reset()
            i_episode += 1

    env.close()

if __name__ == "__main__":
    policies = ["deep_rl_sb3", "deep_rl_rsl"]
    n_episodes = 1

    for policy in policies:
        gs.init(logging_level="warning")
        agent_run(policy, n_episodes)
        gs.destroy()

    print("Evaluation complete")