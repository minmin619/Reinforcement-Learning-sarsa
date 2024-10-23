import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        self.num_tiles_per_dim = np.ceil((self.state_high - self.state_low) / self.tile_width).astype(int) + 1
        self.total_tiles = np.prod(self.num_tiles_per_dim)
        self.tile_offsets = [self.tile_width * i / self.num_tilings for i in range(self.num_tilings)]
    
    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * self.total_tiles

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros(self.feature_vector_len())

        feature_vector = np.zeros(self.feature_vector_len())
        for tiling_idx, offset in enumerate(self.tile_offsets):
            tile_indices = ((s - self.state_low + offset) // self.tile_width).astype(int)
            flat_index = np.ravel_multi_index(tile_indices, self.num_tiles_per_dim)
            feature_index = a * self.num_tilings * self.total_tiles + tiling_idx * self.total_tiles + flat_index
            feature_vector[feature_index] = 1
        return feature_vector
def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    for episode in range(num_episode):
        s = env.reset()
        done = False
        a = epsilon_greedy_policy(s, done, w)
        x = X(s, done, a)
        z = np.zeros(X.feature_vector_len())
        Q_old = 0
        
        while not done:
            s_, r, done, _ = env.step(a)
            a_ = epsilon_greedy_policy(s_, done, w)
            x_ = X(s_, done, a_)
            Q = np.dot(w, x)
            Q_ = np.dot(w, x_)
            delta = r + gamma * Q_ - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_
            x = x_
            a = a_
    
    return w
