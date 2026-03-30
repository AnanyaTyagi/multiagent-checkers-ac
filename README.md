# 🧠 Multi-Agent Reinforcement Learning: 6x6 Checkers

This project implements a custom **PettingZoo AEC environment** for a 6x6 Checkers game and trains an **Actor-Critic agent using self-play**.

---

## 🎮 Environment: 6x6 Checkers

### Overview
- Agents: `["player_0", "player_1"]`
- Board size: `6 x 6`
- Turn-based gameplay (AEC)

This environment follows the PettingZoo Agent Environment Cycle (AEC), where agents act sequentially.

---

## 📦 Observation Space

Each agent observes:

```
{
    "observation": Box(shape=(36,), values in [-2, 2]),
    "action_mask": Box(shape=(144,), values in {0,1})
}
```

- **observation**: flattened 6x6 board  
  - `0`: empty  
  - `1 / 2`: own man / king  
  - `-1 / -2`: opponent man / king  

- **action_mask**: binary vector indicating valid actions  

---

## 🎯 Action Space

```
Discrete(144)
```

Each action encodes:
```
action = (row * 6 + col) * 4 + direction
```

Directions:
- `0`: up-left  
- `1`: up-right  
- `2`: down-left  
- `3`: down-right  

Invalid actions are masked out.

---

## 🧩 Game Rules

- Pieces move diagonally  
- Captures (jumps) are **mandatory**  
- Multi-jump is supported  
- Pieces become **kings** at the opposite side  
- Game ends when:
  - opponent has no pieces, OR  
  - opponent has no legal moves  

---

## 🏆 Rewards

- Win: `+1`  
- Loss: `-1`  
- Capture: `+0.2`  
- Get captured: `-0.2`  
- King promotion: `+0.3`  
- Illegal move: immediate loss  

---

## 🤖 Agent: Actor-Critic (Self-Play)

The agent uses a shared neural network:

- Input: board state (36 features)  
- Actor head: action probabilities  
- Critic head: state value  

### Training
- Self-play (same model controls both agents)  
- Masked action sampling  
- Advantage: `R - V(s)`  
- Loss:
  - Policy gradient loss  
  - Value loss (MSE)  
  - Entropy regularization  

---

## 🔁 Training Loop

```
env.reset()
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = policy(obs)

    env.step(action)
```

---

## ▶️ Running the Project

### Install dependencies
```
pip install numpy gymnasium pettingzoo torch
```

### Run training
```
python myrunner.py
```

---

## 📊 Output

The program:
- trains using self-play  
- prints board states during evaluation  
- outputs final cumulative rewards  

---

## 📁 Project Structure

```
mycheckersenv.py   # Custom PettingZoo environment
myagent.py         # Actor-Critic agent
myrunner.py        # Training and evaluation
README.md          # Documentation
```

---

## 🚀 Summary

This project demonstrates:
- Custom multi-agent environment design (PettingZoo AEC)  
- Actor-Critic with function approximation  
- Self-play training  
- Reinforcement learning in a competitive setting  
