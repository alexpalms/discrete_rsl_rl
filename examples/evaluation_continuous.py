import logging
import sys

import genesis as gs  # pyright:ignore[reportMissingTypeStubs]

from examples.legged_locomotion_continuous.agent.deep_rl_rsl import (
    Agent as DeepRlAgentRSL,
)
from examples.legged_locomotion_continuous.agent.deep_rl_sb3 import (
    Agent as DeepRlAgentSB3,
)
from examples.legged_locomotion_continuous.environment.environment import Environment

logger = logging.getLogger("containerl.environment_client")
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)


def agent_run(policy: str, n_episodes: int):
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
        obs, reward, done, _ = env.step(action)
        total_reward += reward[0]
        if bool(done[0]):
            logger.info(f'Total reward for policy "{policy}": {total_reward}')
            obs, _ = env.reset()
            i_episode += 1

    env.close()


def main():
    policies = ["deep_rl_sb3", "deep_rl_rsl"]
    n_episodes = 1

    for policy in policies:
        gs.init(logging_level="warning")  # pyright:ignore[reportUnknownMemberType]
        agent_run(policy, n_episodes)
        gs.destroy()

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
