# reference: https://github.com/sirine-b/DDPG/blob/main/DDPG_model.ipynb

import os
import math
import random
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from env import ContinuousCartPoleEnv


# 커스텀 환경 초기 설정
env = ContinuousCartPoleEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# 하이퍼파라미터 설정
NUM_EPISODES = 1500
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.01
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
BUFFER_SIZE = 200_000
HIDDEN_LAYERS_ACTOR = 128
HIDDEN_LAYERS_CRITIC = 256

# Best Buffer 설정
BEST_BUFFER_SIZE = 10_000        # how many transitions the best buffer can hold (flattened)
BEST_TOP_EPISODES = 2          # keep up to this many top trajectories
BEST_SAMPLE_RATIO = 0.15        # fraction of each training batch drawn from best buffer
BEST_ACTION_PROB = 0.20         # probability to use a best-buffer action when episode is performing worse than best
BEST_ACTION_NOISE_STD = 0.05    # gaussian noise added to best action when used

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

# 신경망
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_layer_norm=False):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, HIDDEN_LAYERS_ACTOR)
        self.ln1 = nn.LayerNorm(HIDDEN_LAYERS_ACTOR) if use_layer_norm else nn.Identity()
        self.l2 = nn.Linear(HIDDEN_LAYERS_ACTOR, HIDDEN_LAYERS_ACTOR)
        self.ln2 = nn.LayerNorm(HIDDEN_LAYERS_ACTOR) if use_layer_norm else nn.Identity()
        self.l3 = nn.Linear(HIDDEN_LAYERS_ACTOR, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.ln1(self.l1(state)))
        x = torch.relu(self.ln2(self.l2(x)))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_batch_norm=False):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, HIDDEN_LAYERS_CRITIC)
        self.bn1 = nn.BatchNorm1d(HIDDEN_LAYERS_CRITIC) if use_batch_norm else nn.Identity()
        self.l2 = nn.Linear(HIDDEN_LAYERS_CRITIC, HIDDEN_LAYERS_CRITIC)
        self.bn2 = nn.BatchNorm1d(HIDDEN_LAYERS_CRITIC) if use_batch_norm else nn.Identity()
        self.l3 = nn.Linear(HIDDEN_LAYERS_CRITIC, 1)

    def forward(self, state, action):
        x = torch.relu(self.bn1(self.l1(torch.cat([state, action], dim=1))))
        x = torch.relu(self.bn2(self.l2(x)))
        return self.l3(x)



# 리플레이 버퍼 및 노이즈
class ReplayBuffer:
    """Standard simple replay buffer storing transitions as tuples."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class BestTrajectories:
    """
    Keeps the top-K trajectories by cumulative reward and provides a flattened buffer of
    the best transitions (up to capacity). Also supports finding the best action whose
    state is nearest to a given current state (helps contextual borrowing of actions).
    """
    def __init__(self, capacity_transitions=BEST_BUFFER_SIZE, max_trajectories=BEST_TOP_EPISODES):
        self.capacity = int(capacity_transitions)
        self.max_trajectories = max_trajectories
        self.top_trajectories = []
        self._flat_buffer = []

    def add_trajectory(self, trajectory, reward):
        if len(trajectory) == 0:
            return
        self.top_trajectories.append((float(reward), list(trajectory)))
        self.top_trajectories.sort(key=lambda x: x[0], reverse=True)
        self.top_trajectories = self.top_trajectories[: self.max_trajectories]
        flat = []
        for (_, traj) in self.top_trajectories:
            for t in traj:
                if len(flat) < self.capacity:
                    flat.append(t)
                else:
                    break
            if len(flat) >= self.capacity:
                break
        self._flat_buffer = flat

    def sample(self, batch_size):
        if len(self._flat_buffer) == 0:
            raise IndexError("Best buffer is empty")
        if len(self._flat_buffer) >= batch_size:
            batch = random.sample(self._flat_buffer, batch_size)
        else:
            batch = [random.choice(self._flat_buffer) for _ in range(batch_size)]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_one(self):
        if len(self._flat_buffer) == 0:
            raise IndexError("Best buffer is empty")
        return random.choice(self._flat_buffer)

    def sample_nearest_action(self, state):
        """
        Given a current state (1D numpy), find the transition whose state is nearest in L2
        distance and return its action. If the flat buffer is empty, raise IndexError.
        """
        if len(self._flat_buffer) == 0:
            raise IndexError("Best buffer is empty")
        states = np.stack([t[0] for t in self._flat_buffer])
        s = np.array(state).reshape(1, -1)
        dists = np.linalg.norm(states - s, axis=1)
        idx = int(np.argmin(dists))
        _, action, _, _, _ = self._flat_buffer[idx]
        return np.array(action, dtype=np.float32).flatten()

    def __len__(self):
        return len(self._flat_buffer)


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.05, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.dtype = action_space.dtype
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.asarray(np.clip(action + ou_state, self.low, self.high), dtype=self.dtype)



# DDPG 에이전트 정의
class DDPG:
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 use_target=True,
                 use_norm=False,
                 buffer_size=BUFFER_SIZE,
                 best_buffer_size=BEST_BUFFER_SIZE,
                 best_top_episodes=BEST_TOP_EPISODES,
                 best_sample_ratio=BEST_SAMPLE_RATIO,
                 best_action_prob=BEST_ACTION_PROB,
                 best_action_noise_std=BEST_ACTION_NOISE_STD):
        """
        Improved best-buffer behavior:
          - store multiple top trajectories (top-K) instead of only one
          - sample best transitions from a flattened buffer built from top trajectories
          - when borrowing actions for exploration, pick the action whose state is nearest to current state
          - support adaptive best-sample ratio when training stagnates
        """
        self.actor = Actor(state_dim, action_dim, max_action, use_layer_norm=use_norm).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action, use_layer_norm=use_norm).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, action_dim, use_batch_norm=use_norm).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim, use_batch_norm=use_norm).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.best_trajectories = BestTrajectories(best_buffer_size, max_trajectories=best_top_episodes)
        self.best_reward = -float("inf")  # best episode cumulative reward seen so far
        self.best_trajectory = []         # convenience pointer to latest best trajectory

        self.use_target = use_target

        self.base_best_sample_ratio = float(np.clip(best_sample_ratio, 0.0, 1.0))
        self.best_sample_ratio = float(np.clip(best_sample_ratio, 0.0, 1.0))
        self.best_action_prob = float(np.clip(best_action_prob, 0.0, 1.0))
        self.best_action_noise_std = float(best_action_noise_std)

        self.episodes_since_improvement = 0

    def select_action(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()
        return action

    def update_best_buffer(self, trajectory, episode_reward):
        """
        Update best buffer if the given episode trajectory reached a higher cumulative reward.
        Maintains top-K trajectories for robustness and better replay diversity.
        """
        if episode_reward > self.best_reward:
            # New best -> update best reward and add trajectory
            self.best_reward = episode_reward
            self.best_trajectory = list(trajectory)
            self.best_trajectories.add_trajectory(self.best_trajectory, episode_reward)
            self.episodes_since_improvement = 0
            print(f"[BestTrajectories] New best episode reward: {episode_reward:.3f} -> best trajectories updated (flat_size={len(self.best_trajectories)})")
        else:
            self.best_trajectories.add_trajectory(trajectory, episode_reward)
            self.episodes_since_improvement += 1

    def train(self):
        """
        Training step uses a mixture of samples from normal replay buffer and best buffer.
        The fraction from the best buffer is controlled by self.best_sample_ratio, which can adapt
        when training stagnates.
        """

        if len(self.replay_buffer) < 1:
            return

        if self.episodes_since_improvement > 20:
            self.best_sample_ratio = min(0.5, self.best_sample_ratio + 0.01)
        else:
            self.best_sample_ratio = max(self.base_best_sample_ratio, self.best_sample_ratio * 0.995)

        # Determine how many samples to draw from best buffer
        n_best = int(BATCH_SIZE * self.best_sample_ratio)
        n_best = min(n_best, len(self.best_trajectories)) if len(self.best_trajectories) > 0 else 0
        n_general = BATCH_SIZE - n_best

        # If not enough general samples, skip training
        if len(self.replay_buffer) < max(1, n_general):
            return

        state_g, action_g, reward_g, next_state_g, done_g = self.replay_buffer.sample(n_general)

        if n_best > 0:
            try:
                state_b, action_b, reward_b, next_state_b, done_b = self.best_trajectories.sample(n_best)
                state = np.vstack([state_b, state_g])
                action = np.vstack([action_b, action_g])
                reward = np.vstack([reward_b.reshape(-1, 1), reward_g.reshape(-1, 1)])
                next_state = np.vstack([next_state_b, next_state_g])
                done = np.vstack([done_b.reshape(-1, 1), done_g.reshape(-1, 1)])
            except Exception:
                # If the best buffer sampling fails for any reason, fall back to general-only sampling
                state, action, reward, next_state, done = state_g, action_g, reward_g, next_state_g, done_g
        else:
            state, action, reward, next_state, done = state_g, action_g, reward_g, next_state_g, done_g

        state = torch.FloatTensor(state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        reward = torch.FloatTensor(reward.reshape(-1, 1)).to(DEVICE)
        done = torch.FloatTensor(done.reshape(-1, 1)).to(DEVICE)

        if self.use_target:
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                target_Q = self.critic_target(next_state, next_action)
        else:
            with torch.no_grad():
                next_action = self.actor(next_state)
                target_Q = self.critic(next_state, next_action)

        target_Q = reward + (1 - done) * GAMMA * target_Q

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.use_target:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)



# 학습 함수
def train_ddpg(use_target_network=True, use_norm=False, num_episodes=NUM_EPISODES,
               best_sample_ratio=BEST_SAMPLE_RATIO, best_action_prob=BEST_ACTION_PROB,
               best_buffer_size=BEST_BUFFER_SIZE, best_action_noise_std=BEST_ACTION_NOISE_STD,
               best_top_episodes=BEST_TOP_EPISODES):
    """
    Main training function.
    Implements:
      - Episode-based tracking of cumulative return
      - Updating a 'best' replay buffer when a new best episode is found (top-K trajectories)
      - Using actions sampled from the best buffer for exploration when current episode reward is worse than best,
        but choosing the action whose state is most similar to current state (contextual borrowing)
      - Training minibatches that combine samples from the general replay buffer and the best buffer according to best_sample_ratio
      - Adaptive increase of best_sample_ratio when training stagnates
    """
    agent = DDPG(state_dim, action_dim, max_action,
                 use_target=use_target_network, use_norm=use_norm,
                 best_buffer_size=best_buffer_size,
                 best_top_episodes=best_top_episodes,
                 best_sample_ratio=best_sample_ratio,
                 best_action_prob=best_action_prob,
                 best_action_noise_std=best_action_noise_std)

    noise = OUNoise(env.action_space)

    rewards = []
    avg_reward = 0.0

    for episode in range(num_episodes):
        state, _ = env.reset()
        noise.reset()
        episode_reward = 0.0
        done = False
        step = 0

        # store episode transitions to be able to update best buffer at episode end
        episode_trajectory = []

        while not done:
            action_actor = agent.select_action(state)

            # Condition to possibly override action with a best-buffer action:
            # - Only consider if we have a best seen reward
            # - If current episode cumulative so far is worse than best, with probability agent.best_action_prob
            #   pick an action from the best buffer that is contextually similar (nearest-state) and add small gaussian noise
            if len(agent.best_trajectories) > 0 and episode_reward < agent.best_reward:
                if random.random() < agent.best_action_prob:
                    try:
                        best_action = agent.best_trajectories.sample_nearest_action(state)
                        noise_to_add = np.random.randn(*best_action.shape) * agent.best_action_noise_std
                        action_actor = np.clip(best_action + noise_to_add, -max_action, max_action)
                    except Exception:
                        pass

            # Apply OU noise on top of chosen actor action (this preserves action-space limits)
            action = noise.get_action(action_actor, step)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            agent.replay_buffer.push(state.astype(np.float32),
                                     np.array(action, dtype=np.float32),
                                     np.array(reward, dtype=np.float32),
                                     next_state.astype(np.float32),
                                     float(done))

            # store transition in episode trajectory
            episode_trajectory.append((state.astype(np.float32),
                                       np.array(action, dtype=np.float32),
                                       np.array(reward, dtype=np.float32),
                                       next_state.astype(np.float32),
                                       float(done)))

            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.train()

            state = next_state
            episode_reward += float(reward)
            step += 1

        # Episode ended: consider updating best buffer with the whole trajectory
        agent.update_best_buffer(episode_trajectory, episode_reward)

        rewards.append(episode_reward)

        if episode >= 10:
            avg_reward = sum(rewards[episode - 10:episode]) / 10

        if env.vpy_canvas is not None:
            env.vpy_canvas.caption = f'[{episode + 1}/{num_episodes}] avg reward: {avg_reward:.2f} prev reward: {episode_reward:.2f}'

        if (episode + 1) % 10 == 0 or episode == 0:
            if episode >= 500 and avg_reward >= 1500:
                env.render_mode = 'human'
            print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.2f} (Best Reward: {agent.best_reward if agent.best_reward != -float('inf') else 'N/A'})")

    return agent, rewards

if __name__ == "__main__":
    print(f"train started: {datetime.now().isoformat()}")
    agent, rewards = train_ddpg(use_target_network=True, use_norm=False, num_episodes=NUM_EPISODES)
