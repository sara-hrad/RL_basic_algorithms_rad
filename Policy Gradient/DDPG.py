import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Concatenate
import numpy as np
import gym
from datetime import datetime
import matplotlib.pyplot as plt

# Environment
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
log_dir = "log/episode_rewards"
summary_writer = tf.summary.create_file_writer(log_dir)
print(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# print(state_dim)

# Hyperparameters
gamma = 0.99
tau = 0.05
buffer_size = 1000
batch_size = 128
actor_lr = 0.001
critic_lr = 0.001


# Actor and Critic Networks
class Actor(Model):
    def __init__(self, action_dim=1):
        """
        :param action_dim: 1
        """
        super(Actor, self).__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.output_layer = Dense(action_dim, activation='tanh')

    def call(self, state):
        """
        :param state: env.observation
        :return: actions
        """
        # model
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.state_h1 = Dense(128, activation='relu')
        self.action_h1 = Dense(128, activation='relu')
        self.combine_layer = Concatenate()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(1, activation='linear')

    def call(self, state, action):
        """
        :param state: env.observation
        :param action: Actor.call(env.observation)
        :return: Q value
        """
        state_val = self.state_h1(state)
        action_val = self.action_h1(action)
        combined = self.combine_layer([state_val, action_val])
        x = self.dense1(combined)
        x = self.dense2(x)
        return self.output_layer(x)


# DDPG Agent
class DDPGAgent:
    def __init__(self, action_dim=1, actor_lr=0.001, critic_lr=0.001):
        """
        :param action_dim: 1
        :param actor_lr: 0.0001
        :param critic_lr: 0.001
        """
        self.actor = Actor(action_dim)
        self.critic = Critic()
        self.target_actor = Actor(action_dim)
        self.target_critic = Critic()
        self.actor_optimizer = tf.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.optimizers.Adam(critic_lr)
        self.buffer = []

    def remember(self, state, action, reward, next_state, buffer_size):
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > buffer_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def update_target_networks(self, tau):
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()

        target_actor_weights = self.target_actor.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        for i in range(len(target_actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]

        for i in range(len(target_critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

    def train(self, batch_size, tau):
        states, actions, rewards, next_states = self.sample_batch(batch_size)

        # Train Critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor.call(next_states)
            # print(target_actions)
            target_q_values = self.target_critic.call(next_states, target_actions)
            # print(target_q_values)
            y = rewards + gamma * target_q_values
            current_q_values = self.critic.call(states, actions)
            critic_loss = tf.reduce_mean(tf.square(y - current_q_values))
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Train Actor
        with tf.GradientTape() as tape:
            predicted_actions = self.actor.call(states)
            actor_loss = -tf.reduce_mean(self.critic.call(states, predicted_actions))
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # Update Target Networks
        self.update_target_networks(tau)


# Training
agent = DDPGAgent(action_dim, actor_lr, critic_lr)
num_episodes = 1000
for episode in range(1, num_episodes + 1):
    state = env.reset()[0]
    episode_reward = 0

    for t in range(1, 1000):
        action = agent.actor.call(np.reshape(state, [1, state_dim]))
        # action = action.numpy()[0] + np.random.normal(0, 0.1, size=action_dim)
        action = action.numpy()[0]
        action = np.clip(action, -action_bound, action_bound)

        next_state, reward, done, _, _ = env.step(action)

        agent.remember(state, action, reward, next_state, batch_size)
        agent.train(batch_size=batch_size, tau=tau)

        state = next_state
        episode_reward += reward

        if done:
            print("The task is done.")
            break
    print(f"Episode: {episode}, Reward: {episode_reward}")
    with summary_writer.as_default(step=episode):
        tf.summary.scalar('Episode Reward', episode_reward)

