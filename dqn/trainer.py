import numpy as np
import tensorflow as tf
import random
from collections import deque
from dqn.model import *


class Trainer(object):
    def __init__(self, model, sess, batch_size, initial_epsilon, final_epsilon):
        self.model = model
        self.sess = sess
        self.replay_buffer = deque()
        self.batch_size = batch_size
        self.epsilon = initial_epsilon
        self.initial_epsilon =  initial_epsilon
        self.final_epsilon = final_epsilon

    def take_action(self, index, info, env):

        state = info.states[index]
        Q_value = self.sess.run(self.model.Q_value,
            feed_dict = {self.model.state_input:[state]}) [0]

        # TODO: implement discrete action space
        # action domain [0,1,2]
        action = np.argmax(Q_value)
        print(type(action))
        if random.random() <= self.epsilon:
            action = np.array([random.randint(0, 2), random.randint(0, 2)])
            print(type(action))

        self.epsilon -= (self.initial_epsilon - self.final_epsilon)/10000
        new_info = env.step(action)

        new_state = new_info.states[index]
        reward = new_info.rewards[index]
        done = new_info.local_done[index]
        self.add_experience(self, state,  action, reward, new_state, done)

        return new_info

    def add_experience(self, state, action, reward, new_state):
        one_hot_action = np.zeros(self.a_size)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, action, reward, new_state))
        if len(self.replay_buffer) > replay_size:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > batch_size:
            self.train_Q_network()

    def train_Q_network():
        minibatch = random.sample(self.replay_buffer, batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]

        # target_Q = rewrad + gamma * next_Q
        target_Q = []
        next_Q_value_batch = self.sess.run(
            self.model.Q_value, feed_dict={self.model.state_input:next_state_batch})

        for i in range(0, batch_size):
            if done[i]:
                target_Q.append(reward_batch[i])
            else:
                target_Q.append(reward_batch[i] + gamma * np.max(next_Q_value_batch[i]))

        self.model.optimizer.run(feed_dict = {
            self.model.target_Q : target_Q,
            self.model.action_input : action_batch,
            self.model.state_input : state_batch
        })
