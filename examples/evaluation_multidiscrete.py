import logging
import sys
from typing import cast

# pyright: reportUnknownMemberType=false
import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict  # pyright:ignore[reportMissingTypeStubs]

from examples.maintenance_scheduling_multidiscrete.agent.deep_rl_rsl import (
    Agent as DeepRlAgentRSL,
)
from examples.maintenance_scheduling_multidiscrete.agent.deep_rl_sb3 import (
    Agent as DeepRlAgentSB3,
)
from examples.maintenance_scheduling_multidiscrete.agent.no_action_agent import (
    Agent as NoActionAgent,
)
from examples.maintenance_scheduling_multidiscrete.agent.random_agent import (
    Agent as RandomAgent,
)
from examples.maintenance_scheduling_multidiscrete.agent.top_k_failure_probability_heuristic import (
    Agent as TopKFailureProbabilityAgent,
)
from examples.maintenance_scheduling_multidiscrete.agent.top_k_risk_heuristic import (
    Agent as TopKRiskAgent,
)
from examples.maintenance_scheduling_multidiscrete.agent.top_k_score_heuristic import (
    Agent as TopKScoreAgent,
)
from examples.maintenance_scheduling_multidiscrete.environment.environment import (
    Environment,
)

logger = logging.getLogger("containerl.environment_client")
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)


def agent_run(seed: int, policy: str, n_episodes: int) -> list[float]:
    num_envs = 1
    env = Environment(num_envs)
    if policy == "random":
        agent = RandomAgent(env)
    elif policy == "no_action":
        agent = NoActionAgent(env)
    elif policy == "top_k_failure_probability":
        agent = TopKFailureProbabilityAgent(env)
    elif policy == "top_k_score":
        agent = TopKScoreAgent(env)
    elif policy == "top_k_risk":
        agent = TopKRiskAgent(env)
    elif policy == "deep_rl_sb3":
        agent = DeepRlAgentSB3(env)
    elif policy == "deep_rl_rsl":
        agent = DeepRlAgentRSL(env)
    else:
        raise NotImplementedError("Policy not implemented")
    obs, _ = env.reset(seed=seed)
    risk_history: list[float] = []
    cumulative_risk = 0.0
    i_episode = 0
    while i_episode < n_episodes:
        action = agent.get_action(cast(TensorDict, obs["policy"]))
        obs, reward, done, _ = env.step(action)
        cumulative_risk -= float(reward[0].cpu().numpy())
        risk_history.append(cumulative_risk)
        if torch.any(done):
            logger.info("==========================")
            logger.info(f'Final Risk for policy "{policy}": {cumulative_risk}')
            obs, _ = env.reset()
            i_episode += 1
            cumulative_risk = 0.0

    env.close()
    return risk_history


if __name__ == "__main__":
    seed = 42
    policies: list[str] = [
        "no_action",
        "random",
        "top_k_failure_probability",
        "top_k_score",
        "top_k_risk",
        "deep_rl_sb3",
        "deep_rl_rsl",
    ]
    n_episodes = 1

    risk_chart: dict[str, list[float]] = {}

    for policy in policies:
        risk_chart[policy] = agent_run(seed, policy, n_episodes)

    plt.figure(figsize=(10, 6))
    font = {"size": 14}
    plt.rc("font", **font)
    plt.plot(risk_chart["no_action"], label="No Maintenance")
    plt.plot(risk_chart["random"], label="Random Maintenance", linestyle="--")
    plt.plot(
        risk_chart["top_k_failure_probability"],
        label="Top K Failure Probability Maintenance (K=10)",
        color="green",
        marker="o",
    )
    plt.plot(
        risk_chart["top_k_score"],
        label="Top K Score Maintenance (K=10)",
        color="red",
        marker="*",
    )
    plt.plot(
        risk_chart["top_k_risk"],
        label="Top K Risk Maintenance (K=10)",
        color="blue",
        marker="*",
    )
    plt.plot(risk_chart["deep_rl_sb3"], label="Deep RL SB3", color="orange", marker="+")
    plt.plot(risk_chart["deep_rl_rsl"], label="Deep RL RSL", color="purple", marker="x")
    plt.xlabel("Time [days]", fontsize=14, labelpad=10)
    plt.ylabel("Failure Risk Probability []", fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right", fontsize=13)
    plt.title("Failure Risk Over Time", fontsize=16, pad=20)
    plt.show(block=True)
