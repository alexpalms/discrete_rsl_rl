import torch

class Agent:
    def __init__(self, env):
        self.env = env
        self.k = self.env.action_space.shape[0]

    def get_action(self, obs):
        # obs['nodes_infection_probability']: shape (num_sim, n_nodes), torch tensor

        infection_probs = obs[:, -self.env.n:]  # (num_sim, n_nodes)
        score = obs[:, :self.env.n]  # (num_sim, n_nodes)
        risk = infection_probs * score

        # Step 1 & 2 combined: top-k values and their indices, sorted in descending order
        # torch.topk already returns sorted results
        top_k_values, top_k_indices = torch.topk(risk, self.k, dim=1)

        # Step 3: Convert zero-based indices to one-based indexing
        top_k_indices_one_based = top_k_indices + 1

        return top_k_indices_one_based