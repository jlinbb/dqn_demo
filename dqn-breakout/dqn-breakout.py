import cv2
import gym
import tensorflow as tf
import numpy as np
import random
from PIL import Image
from collections import deque
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit()



CNN_INPUT_SIZE = 80
CNN_INPUT_DEPTH = 4  # series length
BATCH_SIZE = 64
FINAL_EPSILON = 0.0001
EPISODE = 1000000
STEP = 1200
ENV = 'Breakout-v4'

# Image processing
class ImageProcess():
    def color2Gray(self, state):
        # pic = Image.fromarray(state)
        # pic.show()
        gray_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        _, binary_state = cv2.threshold(gray_state, 3, 255, cv2.THRESH_BINARY)

        binary_state = cv2.resize(binary_state, (CNN_INPUT_SIZE, 105))
        binary_state = binary_state[25:, :]

        # print(binary_state.shape)
        pic = Image.fromarray(binary_state)
        pic.save()
        return binary_state


# build dqn network
class DQN():
    def __init__(self, env, learning_rate=0.001, epsilon=1, gamma=0.9, buffer_size=20000, log=False):
        self.imageProcess = ImageProcess()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.replay_buffer = deque()

        self.time_step = 0
        self.observe_time = 0

        self.action_dim = env.action_space.n
        self.state_dim = CNN_INPUT_SIZE**2

        self.session = tf.InteractiveSession()
        if log:
            self.log = tf.summary.FileWriter('/logs', self.session.graph)
        self.create_network()
        self.session.run(tf.global_variables_initializer())

    def generate_weights(self, shape):
        weight = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weight)

    def generate_bias(self, shape):
        bias = tf.constant(0.01, shape=shape)
        return bias

    def create_network(self):
        # input size: [batch， in_height，in_width，in_channels]
        self.input_layer = tf.placeholder(tf.float32, [None, CNN_INPUT_SIZE, CNN_INPUT_SIZE, CNN_INPUT_DEPTH])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])

        # conv1 layer
        w1 = self.generate_weights([8, 8, 4, 32])
        b1 = self.generate_bias([32])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, w1, strides=[1, 4, 4, 1], padding='SAME') + b1)

        # conv2 layer
        w2 = self.generate_weights([4, 4, 32, 64])
        b2 = self.generate_bias([64])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, w2, strides=[1, 2, 2,1], padding='SAME') + b2)

        # conv3 layer
        w3 = self.generate_weights([3, 3, 64, 64])
        b3 = self.generate_bias([64])
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 6400])

        # fc1 layer
        fc_w1 = self.generate_weights([6400, 512])
        fc_b1 = self.generate_bias([512])
        fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, fc_w1) + fc_b1)

        # fc2 layer
        fc_w2 = self.generate_weights([512, self.action_dim])
        fc_b2 = self.generate_bias([self.action_dim])
        self.Q_value = tf.matmul(fc1, fc_w2) + fc_b2

        # to get the Q value of the input action
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)

        # loss function
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())

        # print(self.action_dim)
        # action = np.zeros((self.action_dim, 1))
        # action[2][0] = 1
        # print(action)

        # print(sess.run(h_conv3, feed_dict={self.input_layer: state}).shape)
        # print(sess.run(self.Q_value, feed_dict={self.input_layer: state}).shape)
        # print(sess.run(Q_action, feed_dict={self.input_layer: state, self.action_input: action}))

    def train_network(self):
        self.time_step += 1
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [batch[0] for batch in minibatch]
        action_batch = [batch[1] for batch in minibatch]
        reward_batch = [batch[2] for batch in minibatch]
        next_state_batch = [batch[3] for batch in minibatch]
        done_batch = [batch[4] for batch in minibatch]

        # use Bellman function to calculate the Q value of the current state
        y_batch = []
        next_Q_value_batch = self.Q_value.eval(feed_dict={self.input_layer: next_state_batch})

        for i in range(BATCH_SIZE):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(next_Q_value_batch[i]))

        # feed data to the network
        self.optimizer.run(feed_dict={
            self.input_layer: state_batch,
            self.action_input: action_batch,
            self.y_input: y_batch
        })


    def experience_store(self, state, action_index, reward, next_state, done):
        action = np.zeros(self.action_dim)
        action[action_index] = 1
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.observe_time += 1

        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_network()


    def get_action(self, state):
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            action_index = np.argmax(self.Q_value.eval(feed_dict={self.input_layer: [state]})[0])

        if self.epsilon >= FINAL_EPSILON:
            self.epsilon -= (self.epsilon - FINAL_EPSILON)/10000

        return action_index


def main():
    env = gym.make(ENV)
    agent = DQN(env)


    for episode in range(EPISODE):
        state = env.reset()
        state = agent.imageProcess.color2Gray(state)
        state = np.stack((state, state, state, state), axis=2)
        total_reward = 0

        for step in range(STEP):
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(agent.imageProcess.color2Gray(next_state), (80, 80, 1))
            next_state = np.append(next_state, state[:, :, :3], axis=2)
            total_reward += reward
            agent.experience_store(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        print('Episode:', episode, 'Total Point this Episode is:', total_reward)


# Play

# env = gym.make(ENV)
# dqn = DQN(env)
# state = env.reset()
# ip = ImageProcess()
# # cnn_input = np.array([ip.color2Gray(state)]).reshape(CNN_INPUT_SIZE,CNN_INPUT_SIZE, 1)
# # cnn_input = np.concatenate((cnn_input, cnn_input, cnn_input, cnn_input), -1)
# cnn_input = ip.color2Gray(state)
# cnn_input = np.stack((cnn_input, cnn_input, cnn_input, cnn_input), axis=2)
# print(cnn_input.shape)
# cnn_input = np.expand_dims(cnn_input, axis=0)
#
#
# dqn.create_network(cnn_input)
if __name__ == '__main__':
    main()
