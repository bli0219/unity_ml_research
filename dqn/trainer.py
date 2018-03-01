import numpy as np
import tensorflow as tf
import random
from collections import deque
from dqn.model import *


class Trainer(object):
    def __init__(self, model, sess, batch_size, replay_size, initial_epsilon, final_epsilon, gamma):
        self.model = model
        self.sess = sess
        self.replay_buffer = deque()
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.epsilon = initial_epsilon
        self.initial_epsilon =  initial_epsilon
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.count = 0
        self.step = 0

    def take_action(self, index, info, env):

        state = info.states[index]
        Q_value = self.sess.run(self.model.Q_value,
            feed_dict = {self.model.state_input:[state]}) [0]

        action = np.argmax(Q_value)
        # if random.random() <= self.epsilon:
        #     action = random.randint(0, self.model.a_size-1)

        self.epsilon -= (self.initial_epsilon - self.final_epsilon)/10000
        new_info = env.step([[action]])["MyBrain1"]

        new_state = new_info.states[index]
        reward = new_info.rewards[index]
        done = new_info.local_done[index]

        self.step += 1
        if (done):
            print(self.step)
        self.add_experience(state, action, reward, new_state, done)

        return new_info

    def add_experience(self, state, action, reward, new_state, done):

        one_hot_action = np.zeros(self.model.a_size)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, new_state, done))
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > self.batch_size:
            self.train_Q_network()

    def train_Q_network(self):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]
        # target_Q = rewrad + gamma * next_Q
        target_Q = []
        next_Q_value_batch = self.sess.run(
            self.model.Q_value, feed_dict={self.model.state_input:next_state_batch})

        for i in range(0, self.batch_size):
            if done_batch[i]:
                target_Q.append(reward_batch[i])
            else:
                target_Q.append(reward_batch[i] + self.gamma * np.max(next_Q_value_batch[i]))

        if (done_batch[0]):
            print("action: {}".format(action_batch))
            print("action Q: {}".format(next_Q_value_batch))
            print("reward: {}".format(reward_batch))
            print("target_Q: {}".format(target_Q))
            self.count +=1


        self.model.optimizer.run(feed_dict = {
            self.model.target_Q : target_Q,
            self.model.action_input : action_batch,
            self.model.state_input : state_batch
        })

        if (done_batch[0]):
            print("resulting action Q: {}".format(next_Q_value_batch))
