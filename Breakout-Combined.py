"""
Alireza Mirzaei - AI project
An RI model for playing Breakout Atari game
29th June 2023
"""

# %% Imports

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import gym
import cv2
import numpy as np
import random
import os
import datetime
from collections import deque
import random
import time

# %% Settings

# Environment settings
EPISODES = 300

# Exploration settings
epsilon = 1  # starting epsilon
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.1

#  Stats settings
SHOW_PREVIEW = True
RENDER_PREVIEW = 5  # render every x episodes

# %% Environment
env = gym.make("Breakout-v4")

SAMPLE_WIDTH = 84
SAMPLE_HEIGHT = 84

MODEL_NAME = "16x32-"


# %% FILE 1: play.py


# Convert image to greyscale, resize and normalise pixels
def preprocess(screen, width, height, targetWidth, targetHeight):
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = screen[20:300, 0:200]  # crop off score
    screen = cv2.resize(screen, (targetWidth, targetHeight))
    screen = screen.reshape(targetWidth, targetHeight) / 255
    return screen


def main_play():
    env = gym.make("Breakout-v4")

    SAMPLE_WIDTH = 84
    SAMPLE_HEIGHT = 84

    LATEST_WEIGHTS = tf.train.latest_checkpoint("checkpoints")

    agent = Agent(width=SAMPLE_WIDTH, height=SAMPLE_HEIGHT, actions=env.action_space.n)

    agent.model.load_weights(LATEST_WEIGHTS)

    while True:
        currentLives = 5  # starting lives

        # Reset environment and get initial state
        current_state = env.reset()
        current_state = preprocess(
            current_state,
            env.observation_space.shape[0],
            env.observation_space.shape[1],
            SAMPLE_WIDTH,
            SAMPLE_HEIGHT,
        )
        current_state = np.dstack(
            (current_state, current_state, current_state, current_state)
        )

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            action = np.argmax(agent.get_qs(current_state))

            new_state, reward, done, info = env.step(action)

            new_state = preprocess(
                new_state,
                env.observation_space.shape[0],
                env.observation_space.shape[1],
                SAMPLE_WIDTH,
                SAMPLE_HEIGHT,
            )

            new_state = np.dstack(
                (
                    new_state,
                    current_state[:, :, 0],
                    current_state[:, :, 1],
                    current_state[:, :, 2],
                )
            )

            env.render(mode="rgb_array")

            # If life is lost auto fire next ball
            if info["lives"] < currentLives:
                env.step(1)

            current_state = new_state
            currentLives = info["lives"]  # update lives remaining

            time.sleep(1 / 30)  # lock framerate to aprox 30 fps


# %% File2 -> Agent.py -> Agent :)


class Agent:
    def __init__(self, width, height, actions):
        self.learningRate = 0.0025
        self.replayMemorySize = 10_000  # Number of states that are kept for training
        self.minReplayMemSize = (
            1_000  # Min size of replay memory before training starts
        )
        self.batchSize = 32  # How many samples are used for training, was 32
        self.updateEvery = 10  # Number of batches between updating target network
        self.discount = 0.99  # measure of how much we care about future reward over immediate reward
        self.actions = actions

        self.model = Model(width, height, self.actions)  # model to be trained
        self.model.compile(
            tf.keras.optimizers.Adam(
                learning_rate=self.learningRate,
                beta_1=EPSILON_DECAY,
                beta_2=EPSILON_DECAY,
                epsilon=epsilon,
                amsgrad=False,
            ),
            loss=tf.keras.losses.Huber(),
            metrics=["accuracy"],
        )

        self.targetModel = Model(width, height, self.actions)  # model for predictions
        self.targetModel.set_weights(self.model.get_weights())

        self.replayMemory = deque(maxlen=self.replayMemorySize)

        self.targetUpdateCounter = 0  # counter since last target model update

    # Queries main network for Q values given current state
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *state.shape))[0]

    # Add new step to replayMemory
    def update_replay_memory(self, transition):
        self.replayMemory.append(transition)

    # Clip reward so it is between -1 and 1
    def clip_reward(self, reward):
        if reward < -1:
            reward = -1
        elif reward > 1:
            reward = 1
        return reward

    # replay memory length is over min size
    def over_min_batch_size(self):
        return len(self.replayMemory) >= self.minReplayMemSize

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replayMemory) < self.minReplayMemSize:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replayMemory, self.batchSize)

        # Get current states from minibatch, then query NN model for Q values
        currentStates = np.array([transition[0] for transition in minibatch])

        # Get future states from minibatch, then query NN model for Q values
        newCurrentStates = np.array([transition[3] for transition in minibatch])
        futureQsList = self.targetModel.predict(newCurrentStates)

        batchTargets = np.zeros((self.batchSize, self.actions))

        y = []

        # Enumerate minibatch to prepare for fitting
        for index, (currentState, action, reward, newCurrentState, done) in enumerate(
            minibatch
        ):
            # If not a terminal state, get new q from future states, otherwise set it to reward
            # life lost if reward is less than 0 (treating it as terminal)
            if not done and reward >= 0:
                max_future_q = np.max(futureQsList[index])
                newQ = reward + (self.discount * max_future_q)
            else:
                newQ = reward

            # Update Q value for given state
            action[np.argmax(action)] = newQ

            batchTargets[index][np.argmax(action)] = 1

            # And append to our training data
            y.append(action)

        # Fit on all minibatch and return loss and accuracy
        metrics = self.model.fit(
            currentStates,
            np.array(y),
            batch_size=self.batchSize,
            verbose=0,
            shuffle=False,
        )

        # Update target network counter every episode
        if terminal_state:
            self.targetUpdateCounter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.targetUpdateCounter > self.updateEvery:
            self.targetModel.set_weights(self.model.get_weights())
            self.targetUpdateCounter = 0

        return metrics


# %% File3 -> Model.py -> CNN model


class Model(Model):
    def __init__(self, width, height, actions):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0025,
            beta_1=EPSILON_DECAY,
            beta_2=EPSILON_DECAY,
            epsilon=epsilon,
            amsgrad=False,
        )

        self.conv1 = tf.keras.layers.Conv2D(
            16, [8, 8], strides=4, input_shape=(width, height, 4), activation="relu"
        )

        self.conv2 = tf.keras.layers.Conv2D(32, [4, 4], strides=2, activation="relu")

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(actions, activation="linear")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


# %% File4 -> Breakout.py -> main driver code
# Settings moved to beginning of file


# Convert image to greyscale, resize and normalise pixels
def preprocess(screen, width, height, targetWidth, targetHeight):
    # plt.imshow(screen)
    # plt.show()
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = screen[20:300, 0:200]  # crop off score
    screen = cv2.resize(screen, (targetWidth, targetHeight))
    screen = screen.reshape(targetWidth, targetHeight) / 255
    # plt.imshow(np.array(np.squeeze(screen)), cmap='gray')
    # plt.show()
    return screen


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/" + MODEL_NAME + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
checkpoint_path = "checkpoints/cp-{episode:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

agent = Agent(width=SAMPLE_WIDTH, height=SAMPLE_HEIGHT, actions=env.action_space.n)
agent.model.load_weights(tf.train.latest_checkpoint('checkpoints'))  # Uncomment to load from checkpoint

average_reward = []

# Iterate over episodes

# First set the per-episode logging to false
tf.keras.utils.disable_interactive_logging()

for episode in tqdm(
    range(EPISODES),
    ascii=True,
    unit="episodes",
    ncols=80,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
):
    average_loss = []
    average_accuracy = []
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    current_state = preprocess(
        current_state,
        env.observation_space.shape[0],
        env.observation_space.shape[1],
        SAMPLE_WIDTH,
        SAMPLE_HEIGHT,
    )

    currentLives = 5  # starting lives for episode

    current_state = np.dstack(
        (current_state, current_state, current_state, current_state)
    )

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # Get action from Q table
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        # Get random action
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        episode_reward += reward

        # If life is lost then give negative reward
        if info["lives"] < currentLives:
            reward = -1
            done = True

        reward = agent.clip_reward(reward)

        new_state = preprocess(
            new_state,
            env.observation_space.shape[0],
            env.observation_space.shape[1],
            SAMPLE_WIDTH,
            SAMPLE_HEIGHT,
        )

        new_state = np.dstack(
            (
                new_state,
                current_state[:, :, 0],
                current_state[:, :, 1],
                current_state[:, :, 2],
            )
        )

        if SHOW_PREVIEW and episode % RENDER_PREVIEW == 0:
            env.render(mode="rgb_array")

        # Every step we update replay memory and train main network
        # print(np.dstack((stateStack[0], stateStack[1], stateStack[2], stateStack[3])).shape)
        agent.update_replay_memory(
            (current_state, agent.get_qs(current_state), reward, new_state, done)
        )
        metrics = agent.train(done, step)

        if metrics is not None:
            average_loss.append(metrics.history["loss"][0])
            average_accuracy.append(metrics.history["accuracy"][0])

        current_state = new_state
        currentLives = info["lives"]  # update lives remaining
        step += 1

    if len(average_reward) >= 5:
        average_reward.pop(0)
        average_reward.append(episode_reward)
    else:
        average_reward.append(episode_reward)

    with train_summary_writer.as_default():
        tf.summary.scalar("episode score", episode_reward, step=episode)
        tf.summary.scalar(
            "average score", sum(average_reward) / len(average_reward), step=episode
        )
        tf.summary.scalar("epsilon", epsilon, step=episode)
        if len(average_loss) > 0:
            tf.summary.scalar(
                "loss", sum(average_loss) / len(average_loss), step=episode
            )
        if len(average_accuracy) > 0:
            tf.summary.scalar(
                "accuracy", sum(average_accuracy) / len(average_accuracy), step=episode
            )

    agent.model.save_weights(checkpoint_path.format(episode=episode))

    # Decay epsilon. Only start when replay memory is over min size
    if epsilon > MIN_EPSILON and agent.over_min_batch_size:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

agent.model.save_weights("models/" + MODEL_NAME + current_time, save_format="tf")
