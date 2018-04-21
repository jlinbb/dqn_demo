from collections import deque
import tensorflow as tf
import numpy as np
import gym


INITIAL_EPSILON = 0.5
REPLAY_SIZE = 2000
BATCH_SIZE = 128
FINAL_EPSILON = 0.01
GAMMA = 0.9


class DQN():
    def __init__(self, env):

        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, name=name)

    def create_Q_network(self):
        self.state_input = tf.placeholder("float", [None, self.state_dim])

        W1 = self.weight_variable([self.state_dim, 20], name='w1')
        b1 = self.bias_variable([20], name='b1')

        hidden_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)

        W2 = self.weight_variable([20, self.action_dim], name='W2')
        b2 = self.bias_variable([self.action_dim], name='b2')

        self.Q_value = tf.matmul(hidden_layer, W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    def train_Q_network(self):
        self.time_step += 1
        minibatch_index = np.random.choice(range(len(self.replay_buffer)), BATCH_SIZE)
        minibatch = np.array(self.replay_buffer)[minibatch_index]
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if (len(self.replay_buffer) > REPLAY_SIZE):
            self.replay_buffer.popleft()
        if (len(self.replay_buffer) > BATCH_SIZE):
            self.train_Q_network()

    def egreedy_action(self, state):
        Q_value= self.Q_value.eval(feed_dict = {self.state_input: [state]})[0]
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)


    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input: [state]
        })[0])


ENV_NAME = 'CartPole-v0'
EPISODE = 100000
STEP = 300
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        state = env.reset()
        for step in range(STEP):
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode', episode, 'Average reward:', ave_reward)
            if ave_reward >= 200:
                break


if __name__ == '__main__':
    main()