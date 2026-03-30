import random
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim

import mycheckersenv
from myagent import ActorCritic, obs_to_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def train_self_play(
    episodes=500,
    gamma=0.99,
    lr=1e-3,
    entropy_coef=0.01,
    value_coef=0.5,
    print_every=50,
):
    env = mycheckersenv.env(render_mode=None)
    model = ActorCritic(obs_dim=36, act_dim=144).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_reward_history = []

    for episode in range(1, episodes + 1):
        env.reset()

        # store per-agent trajectories
        trajectories = {
            "player_0": {"obs": [], "masks": [], "actions": [], "log_probs": [], "values": [], "rewards": []},
            "player_1": {"obs": [], "masks": [], "actions": [], "log_probs": [], "values": [], "rewards": []},
        }

        cumulative_rewards = defaultdict(float)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # reward returned by env.last() belongs to current agent for this turn
            cumulative_rewards[agent] += reward

            if len(trajectories[agent]["values"]) > len(trajectories[agent]["rewards"]):
                trajectories[agent]["rewards"].append(float(reward))

            if termination or truncation:
                action = None
            else:
                obs_tensor, mask_tensor = obs_to_tensor(obs, DEVICE)

                if mask_tensor.sum().item() == 0:
                    action = None
                else:
                    action, log_prob, value, entropy = model.act(obs_tensor, mask_tensor)

                    trajectories[agent]["obs"].append(obs_tensor)
                    trajectories[agent]["masks"].append(mask_tensor)
                    trajectories[agent]["actions"].append(action)
                    trajectories[agent]["log_probs"].append(log_prob)
                    trajectories[agent]["values"].append(value.squeeze())

            env.step(action)

        # pad missing rewards if needed
        for agent in ["player_0", "player_1"]:
            while len(trajectories[agent]["rewards"]) < len(trajectories[agent]["values"]):
                trajectories[agent]["rewards"].append(0.0)

        optimizer.zero_grad()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        total_steps = 0

        for agent in ["player_0", "player_1"]:
            rewards = trajectories[agent]["rewards"]
            if len(rewards) == 0:
                continue

            returns = compute_returns(rewards, gamma=gamma)
            returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

            values = torch.stack(trajectories[agent]["values"])
            log_probs = torch.stack(trajectories[agent]["log_probs"])

            advantages = returns - values

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = (advantages ** 2).mean()

            # recompute entropy batch
            obs_batch = torch.stack(trajectories[agent]["obs"])
            mask_batch = torch.stack(trajectories[agent]["masks"])
            action_batch = torch.tensor(
                trajectories[agent]["actions"], dtype=torch.long, device=DEVICE
            )
            _, _, entropy = model.evaluate_actions(obs_batch, action_batch, mask_batch)
            entropy_loss = -entropy.mean()

            total_actor_loss = total_actor_loss + actor_loss
            total_critic_loss = total_critic_loss + critic_loss
            total_entropy_loss = total_entropy_loss + entropy_loss
            total_steps += len(rewards)

        if total_steps > 0:
            loss = total_actor_loss + value_coef * total_critic_loss + entropy_coef * total_entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        episode_total = cumulative_rewards["player_0"] + cumulative_rewards["player_1"]
        episode_reward_history.append(episode_total)

        if episode % print_every == 0:
            recent_avg = np.mean(episode_reward_history[-print_every:])
            print(f"Episode {episode:4d} | recent avg cumulative reward: {recent_avg:.4f}")

    env.close()
    return model


def choose_greedy_action(model, obs):
    obs_tensor, mask_tensor = obs_to_tensor(obs, DEVICE)

    if mask_tensor.sum().item() == 0:
        return None

    with torch.no_grad():
        logits, _ = model.forward(obs_tensor)
        logits = logits.squeeze(0)
        masked_logits = logits.masked_fill(mask_tensor == 0, -1e9)
        action = torch.argmax(masked_logits).item()
    return int(action)


def run_sample_game(model, seed=123):
    env = mycheckersenv.env(render_mode="human")
    env.reset(seed=seed)

    final_cumulative_rewards = {"player_0": 0.0, "player_1": 0.0}

    print("=" * 50)
    print("SAMPLE RUN WITH TRAINED AGENT")
    print("=" * 50)

    turn_idx = 0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        final_cumulative_rewards[agent] += reward

        print(f"Turn {turn_idx} | Agent: {agent} | Reward from last(): {reward:.3f}")

        if termination or truncation:
            action = None
        else:
            action = choose_greedy_action(model, obs)

        env.step(action)
        turn_idx += 1

    print("=" * 50)
    print("FINAL CUMULATIVE REWARDS")
    print(final_cumulative_rewards)
    print("=" * 50)

    env.close()
    return final_cumulative_rewards


def main():
    set_seed(42)

    model = train_self_play(
        episodes=500,
        gamma=0.99,
        lr=1e-3,
        entropy_coef=0.01,
        value_coef=0.5,
        print_every=50,
    )

    torch.save(model.state_dict(), "checkers_ac.pt")
    print("Saved model to checkers_ac.pt")

    final_rewards = run_sample_game(model, seed=999)
    print("Sample game complete.")
    print("Final cumulative reward:", final_rewards)


if __name__ == "__main__":
    main()