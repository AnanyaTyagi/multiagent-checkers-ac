import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=36, act_dim=144, hidden_dim=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """
        obs: tensor of shape [batch, obs_dim] or [obs_dim]
        returns:
            logits: [batch, act_dim]
            value:  [batch, 1]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        x = self.shared(obs)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def act(self, obs, action_mask):
        logits, value = self.forward(obs)
        logits = logits.squeeze(0)
        value = value.squeeze(0)

        masked_logits = logits.masked_fill(action_mask == 0, -1e9)
        dist = Categorical(logits=masked_logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return int(action.item()), log_prob, value, entropy

    def evaluate_actions(self, obs_batch, action_batch, action_mask_batch):
        logits, values = self.forward(obs_batch)
        masked_logits = logits.masked_fill(action_mask_batch == 0, -1e9)
        dist = Categorical(logits=masked_logits)

        log_probs = dist.log_prob(action_batch)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


def obs_to_tensor(obs_dict, device):
    obs = torch.tensor(obs_dict["observation"], dtype=torch.float32, device=device)
    mask = torch.tensor(obs_dict["action_mask"], dtype=torch.bool, device=device)
    return obs, mask