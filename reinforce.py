from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def __call__(self, s) -> int:
        s = torch.FloatTensor(s) if isinstance(s, np.ndarray) else s
        probs = self.model(s)
        action = torch.multinomial(probs, 1).item()
        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.optimizer.zero_grad()
        s = torch.FloatTensor(s) if isinstance(s, np.ndarray) else s
        probs = self.model(s)
        m = torch.distributions.Categorical(probs)
        loss = -m.log_prob(torch.tensor(a)) * gamma_t * delta
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self, b):
        self.b = b

    def __call__(self, s) -> float:
        return self.b

    def update(self, s, G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def __call__(self, s) -> float:
        s = torch.FloatTensor(s) if isinstance(s, np.ndarray) else s
        return self.model(s).item()

    def update(self, s, G):
        s = torch.FloatTensor(s) if isinstance(s, np.ndarray) else s
        G = torch.tensor(G)
        self.optimizer.zero_grad()
        value = self.model(s)
        loss = nn.MSELoss()(value, G)
        loss.backward()
        self.optimizer.step()

def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    G_0_list = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_data = []
        done = False
        while not done:
            action = pi(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        G = 0
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = reward + gamma * G
            if V:
                delta = G - V(state)
                V.update(state, G)
            else:
                delta = G

            pi.update(state, action, gamma ** t, delta)

        G_0_list.append(episode_data[0][2])

    return G_0_list
