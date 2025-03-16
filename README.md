# **SARSA & Reinforcement Learning Implementation**

##  **Overview**
This repository implements **SARSA (State-Action-Reward-State-Action)**, a classic on-policy reinforcement learning algorithm, to solve Markov Decision Processes (MDPs) and optimize decision-making in an environment. The project explores reinforcement learning principles and demonstrates how SARSA can be used to train an agent to navigate a gridworld-like environment.

### **Why SARSA?**
- **On-policy learning**: Updates the action-value function based on the current policy.
- **Exploration-exploitation balance**: Uses ε-greedy policy to explore better actions.
- **Stable learning**: Unlike Q-learning, SARSA considers the next action when updating Q-values, leading to smoother training.

### **SARSA Algorithm**
SARSA is an **on-policy temporal difference (TD) control method** used to find the optimal Q-value function \( Q(s, a) \) for an agent interacting with an environment.

#### **Update Rule:**
```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) +\alpha \left(r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right)
```

Where:
- \( \alpha \) is the learning rate
- \( \gamma \) is the discount factor
- \( r_t \) is the reward at time step \( t \)
- \( Q(s_t, a_t) \) is the action-value function
- \( a_{t+1} \) is the next action chosen using the current policy (on-policy update)
