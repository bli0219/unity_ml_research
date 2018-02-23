
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'IPComplete.greedy=True')


# ## Setup

# In[2]:


import numpy as np
import os
import tensorflow as tf

from unityagents import *
from dqn.model import *
from dqn.trainer import *

num_episode = 100
max_steps = 5e5
batch_size = 16
initial_epsilon = 0.5
final_epsilon = 0.01
train_model = True
summary_freq = 1e4
save_freq = 5e4
env_name = "external"
steps = 0



# ## Training

# In[3]:


print(0)
env = UnityEnvironment(file_name=env_name)
print(0.5)

brain_name = env.external_brain_names[0]
brain = env.brains[brain_name]
#saver = tf.train.Saver()
agent = DQN(brain)
init = tf.global_variables_initializer()

print(1)
 
with tf.Session() as sess:
    # Instantiate model parameters
    sess.run(init)   
    print(1.5)

    info = env.reset(train_mode=train_model)[brain_name] # info.states.shape = [1,8]
    trainer = Trainer(agent, sess, batch_size, initial_epsilon, final_epsilon)
    print(2)

    while steps <= max_steps:
       
        if env.global_done:
            info = env.reset(train_mode=train_model)
        
        if (info.rewards[0] == -99 ): # index for agent
            print("Player controlling agent...")
        else:
            new_info = trainer.take_action(0, info, env) # take_action also adds experience
            trainer.add_experience(info, new_info, action, reward)
            info = new_info
        
        steps += 1

env.close()

