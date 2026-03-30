🧠 Multi-Agent Reinforcement Learning: 6x6 Checkers

This project implements a custom PettingZoo AEC environment for a 6x6 Checkers game and trains an Actor-Critic agent using self-play.

The environment follows the PettingZoo API for multi-agent reinforcement learning, where agents act sequentially using the Agent Environment Cycle (AEC) model.

🎮 Environment: 6x6 Checkers
Overview

This is a 2-player, turn-based board game where agents move diagonally and capture opponent pieces.

Agents: ["player_0", "player_1"]
Board size: 6 x 6
Turn-based gameplay (AEC)
📦 Observation Space

Each agent observes:

{
    "observation": Box(shape=(36,), values in [-2, 2]),
    "action_mask": Box(shape=(144,), values in {0,1})
}
observation: flattened 6x6 board
0: empty
1 / 2: own man / king
-1 / -2: opponent man / king
action_mask: binary vector indicating valid moves

This follows the PettingZoo pattern where legal actions are provided via an action mask.

🎯 Action Space
Discrete(144)

Each action encodes:

a board position (row, col)
a diagonal direction
action = (row * 6 + col) * 4 + direction

Directions:

0: up-left
1: up-right
2: down-left
3: down-right

Invalid actions are masked out.

🧩 Game Rules
Pieces move diagonally
Captures (jumps) are mandatory
Multi-jump is supported (same piece continues)
Pieces become kings at the opposite end
Game ends when:
opponent has no pieces, OR
opponent has no legal moves
🏆 Rewards
Win: +1
Loss: -1
Capture: +0.2
Get captured: -0.2
King promotion: +0.3
Illegal move: immediate loss

Rewards are primarily sparse (terminal), similar to classic PettingZoo games.

🤖 Agent: Actor-Critic (Self-Play)

The agent uses a shared neural network:

Input: board state (36 features)
Actor head: action probabilities
Critic head: state value
Training
Self-play (same model controls both agents)
Masked action sampling
Advantage = R - V(s)
Loss:
Policy gradient loss
Value (MSE) loss
Entropy regularization
🔁 Training Loop

The environment is interacted with using the AEC API:

env.reset()
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    action = policy(obs) if not done else None
    env.step(action)

This follows the standard PettingZoo interaction pattern.

▶️ Running the Project
Install dependencies
pip install numpy gymnasium pettingzoo torch
Run training
python myrunner.py
📊 Sample Output

The runner:

trains via self-play
prints board states during evaluation
outputs final cumulative rewards
📁 Project Structure
mycheckersenv.py   # Custom PettingZoo environment
myagent.py         # Actor-Critic model
myrunner.py        # Training + evaluation
README.md          # Documentation
🚀 Summary

This project demonstrates:

Custom multi-agent environment design (PettingZoo AEC)
Function approximation with neural networks
Actor-Critic reinforcement learning
Self-play training in competitive environments
