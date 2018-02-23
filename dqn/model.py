import tensorflow as tf
import numpy as np


def save_model(sess, saver, model_path="./", steps=0):
    """
    :param steps: Current number of steps in training process.
    :param saver: Tensorflow saver for session.
    """
    last_checkpoint = model_path + '/model-' + str(steps) + '.cptk'
    saver.save(sess, last_checkpoint)
    tf.train.write_graph(sess.graph_def, model_path, 'raw_graph_def.pb', as_text=False)
    print("Saved Model")

def export_graph(model_path, env_name="env", target_nodes="action,value_estimate,action_probs"):
    """
    Exports latest saved model to .bytes format for Unity embedding.
    :param model_path: path of model checkpoints.
    :param env_name: Name of associated Learning Environment.
    :param target_nodes: Comma separated string of needed output nodes for embedded graph.
    """
    ckpt = tf.train.get_checkpoint_state(model_path)
    freeze_graph.freeze_graph(input_graph=model_path + '/raw_graph_def.pb',
                              input_binary=True,
                              input_checkpoint=ckpt.model_checkpoint_path,
                              output_node_names=target_nodes,
                              output_graph=model_path + '/' + env_name + '.bytes',
                              clear_devices=True, initializer_nodes="", input_saver="",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")


class DQN():

    def __init__(self, brain):

        self.s_size = brain.state_space_size
        self.a_size = brain.action_space_size

        self.create_Q_network()
        self.create_training_method()

    def create_Q_network(self):
        W1 = self.weight_variable([self.s_size, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.a_size])
        b2 = self.bias_variable([self.a_size])

        # input layer; using minibatch
        self.state_input = tf.placeholder(
            dtype=tf.float32, shape=[None, self.s_size])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.Q_value = tf.matmul(h_layer, W2) + b2

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.a_size]) # one hot presentation
        self.target_Q = tf.placeholder("float",[None])
        self.Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.target_Q - self.Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
