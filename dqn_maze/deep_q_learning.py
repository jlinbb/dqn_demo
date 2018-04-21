import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DeepQLearning:
    def __init__(self, n_actions,
                 n_features,
                 learning_rate=0.01,
                 discount=0.9,
                 e_greedy=0.1,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 output_graph=False):

        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = discount
        self.epsilon = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.output_graph = output_graph

        self.learning_steps = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # [s, a ,r ,s_]

        self.construct_network()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_eval_net')

        # Q_eval_net -> Q_target_net
        with tf.variable_scope('target_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if self.output_graph:
            tf.summary.FileWriter("logs", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def construct_network(self):
        with tf.variable_scope('input'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='state')
            self.a = tf.placeholder(tf.int32, [None, ], name='actions')
            self.r = tf.placeholder(tf.float32, [None, ], name='reward')
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='state_')
            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('Q_eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='e2')

        with tf.variable_scope('Q_target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name ='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('Q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1)
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('Q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_by_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_by_a, name='error'))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def store_transition(self, s, a, r,s_):
        if not hasattr(self, 'memory_count'):
            self.memory_count = 0

        transition = np.hstack((s, [a, r], s_))
        index = self.memory_count % self.memory_size
        self.memory[index, :] = transition
        self.memory_count += 1

    def choose_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
            action = np.argmax(action_value)
        return action

    def learn(self):
        if self.learning_steps % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\nreplace tartget net params')
        if self.memory_count > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_count, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        _, _ = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:]
            }
        )

        self.learning_steps += 1