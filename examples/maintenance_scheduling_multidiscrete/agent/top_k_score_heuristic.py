import torch

class Agent:
    def __init__(self, env):
        self.env = env
        self.k  = 10
        self.action_space_size = env.action_space.shape[0]
        assert self.env.action_space.shape[0] <= self.k, f"Action space size ({env.action_space.shape[0]}) must be <= top k {self.k}"

    def get_action(self, obs):
        score = obs[:, :self.env.n]  # (num_sim, n_nodes)

        # Step 1 & 2 combined: top-k values and their indices, sorted in descending order
        # torch.topk already returns sorted results
        top_k_values, top_k_indices = torch.topk(score, self.k, dim=1)

        # Step 3: Convert zero-based indices to one-based indexing
        top_k_indices_one_based = top_k_indices + 1
        top_k_indices_one_based = torch.cat(
            [
                top_k_indices_one_based,
                top_k_indices_one_based,
            ],
            axis=-1,
        )

        idx = self.env.clock_step % self.k

        return top_k_indices_one_based[:, idx:idx + self.action_space_size]