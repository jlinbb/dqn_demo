import os
from model import build_network
from gym_env import GymEnvironment
from keras import backend as K
import threading
import tensorflow as tf
import random
import numpy as np
import time
import gym


flags = tf.app.flags
flags.DEFINE_string('experiment', 'dqn_breakout', 'Name of current experiment')
flags.DEFINE_string('game', 'Breakout-v0', 'Name of Atari game')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent actor(threads)')
flags.DEFINE_integer('tmax', 800000000, 'Number of training timesteps')
flags.DEFINE_integer('resized_width', 84, 'Screen width')
flags.DEFINE_integer('resized_height', 84, 'Screen height')
flags.DEFINE_integer('agent_history_length', 4, '4 recent frames are used to train the model')
flags.DEFINE_integer('network_update_frequency', 32, 'Frequency of updating each actor thread')
flags.DEFINE_integer('target_network_update_frequency', 10000, 'Frequency of reseting the target network')
flags.DEFINE_float('learning_rate', 0.0001, 'Initail learning rate')
flags.DEFINE_float('gamma', 0.95, 'Reward discount factor')
flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, '# of timesteps to anneal epsilon')
flags.DEFINE_string('summary_dir', 'saved/summaries', 'Dir of summary files')
flags.DEFINE_string('checkpoint_dir', 'saved/checkpoints', 'Dir of saved checkpoint files')
flags.DEFINE_integer('summary_interval', 5, '# seconds of saving the summary file')
flags.DEFINE_integer('checkpoint_interval', 600, '# seconds of saving the checkpoint file')
flags.DEFINE_boolean('show_training', True, 'Whether to show the training process')
flags.DEFINE_boolean('testing', False, 'Whether to run the evaluation')
flags.DEFINE_string('checkpoint_path', 'path/to/recent.ckpt', 'Path of recent checkpoint used for evaluation')
flags.DEFINE_string('eval_dir', 'saved/', 'Dir of evaluation files')
flags.DEFINE_integer('num_eval_episodes', 100, '# of evaluation episodes')
# flags.DEFINE_string('reload_path', 'saved/checkpoint/checkpoint', 'reload path')
FLAGS = flags.FLAGS
T, TMAX = 0, FLAGS.tmax


def sample_final_epsilon():
    final_epsilons_list = np.array([.1, .01, .3])
    proba_list = np.array([0.4, 0.4, 0.2])
    return np.random.choice(final_epsilons_list, 1, p=list(proba_list))[0]


def get_num_actions():
    env = gym.make(FLAGS.game)
    # action 0 has no use
    num_actions = 3 if FLAGS.game == "Breakout-v0" else env.action_space.n
    return num_actions


def agent_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver, reload=False):
    global TMAX, T
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]
    summary_placeholders, update_ops, summary_op = summary_ops
    env = GymEnvironment(gym_env=env, width=FLAGS.resized_width, height=FLAGS.resized_height, agent_history_length=FLAGS.agent_history_length)
    s_batch, a_batch, y_batch = [], [], []
    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0.7
    epsilon = 0.7
    if reload:
        epsilon = final_epsilon
    print("Agent ID:", thread_id, "Final epsilon:", final_epsilon)

    time.sleep(2*thread_id)
    t = 0
    while T < TMAX:
        s_t = env.get_initial_state()
        done = False
        ep_reward, episode_ave_max_q, ep_t = 0, 0, 0

        while True:
            readout_t = q_values.eval(session=session, feed_dict = {s: [s_t]})

            a_t = np.zeros([num_actions])
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / FLAGS.anneal_epsilon_timesteps
    
            s_t1, r_t, done, info = env.step(action_index)

            readout_j1 = target_q_values.eval(session = session, feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)

            if done:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + FLAGS.gamma * np.max(readout_j1))
    
            a_batch.append(a_t)
            s_batch.append(s_t)
    
            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            if T % FLAGS.target_network_update_frequency == 0:
                session.run(reset_target_network_params)
    
            if t % FLAGS.network_update_frequency == 0 or done:
                if s_batch:
                    session.run(grad_update, feed_dict={y: y_batch, a: a_batch, s: s_batch})
                # Clear gradients
                s_batch, a_batch, y_batch = [], [], []
    
            if t % FLAGS.checkpoint_interval == 0:
                saver.save(session, FLAGS.checkpoint_dir+"/"+FLAGS.experiment+".ckpt", global_step = t)
    
            if done:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
                print("THREAD:", thread_id,
                      "-- TIME", T,
                      "-- TIMESTEP", t,
                      "-- EPSILON", epsilon,
                      "-- REWARD", ep_reward,
                      "-- Q_MAX %.4f" % (episode_ave_max_q/float(ep_t)))
                break


def build_graph(num_actions):
    s, q_network = build_network(
        num_actions=num_actions,
        agent_history_length=FLAGS.agent_history_length,
        width=FLAGS.resized_width,
        height=FLAGS.resized_height,
        name_scope="q-network"
    )
    network_params = q_network.trainable_weights
    q_values = q_network(s)

    st, target_q_network = build_network(
        num_actions=num_actions,
        agent_history_length=FLAGS.agent_history_length,
        width=FLAGS.resized_width,
        height=FLAGS.resized_height,
        name_scope="target-network"
    )
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(st)

    reset_target_network_params = [
        target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))
    ]
    
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - action_q_values))
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s" : s, 
                 "q_values" : q_values,
                 "st" : st, 
                 "target_q_values" : target_q_values,
                 "reset_target_network_params" : reset_target_network_params,
                 "a" : a,
                 "y" : y,
                 "grad_update" : grad_update}

    return graph_ops


def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Episode_Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Max_Q_Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    tf.summary.scalar("Epsilon", logged_epsilon)
    logged_T = tf.Variable(0.)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op


def train(session, graph_ops, num_actions, saver, saved_path=None):
    envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]
    
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    session.run(tf.global_variables_initializer())
    if saved_path:
        saver.restore(session, saved_path)
    session.run(graph_ops["reset_target_network_params"])
    summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment
    writer = tf.summary.FileWriter(summary_save_path, session.graph)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    reload = saved_path is not None

    if FLAGS.num_concurrent==1:
        agent_thread(0, envs[0], session, graph_ops, num_actions, summary_ops, saver)
    else:
        agents_threads = [threading.Thread(target=agent_thread, args=(thread_id, envs[thread_id], session, graph_ops, num_actions, summary_ops, saver, reload)) for thread_id in range(FLAGS.num_concurrent)]
        for t in agents_threads:
            t.start()

    last_summary_time = 0
    while True:
        if FLAGS.show_training:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > FLAGS.summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now


def test(session, graph_ops, saver):
    saver.restore(session, FLAGS.checkpoint_path)
    monitor_env = gym.make(FLAGS.game)
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    env = GymEnvironment(gym_env=monitor_env, width=FLAGS.resized_width, height=FLAGS.resized_height, agent_history_length=FLAGS.agent_history_length)

    for i_episode in range(FLAGS.num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        done = False
        while not done:
            env.render()
            readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})

            action_index = np.argmax(readout_t)
            s_t1, r_t, done, info = env.step(action_index)
            print("action:", action_index, '-- Done:', done)
            s_t = s_t1
            ep_reward += r_t


def main(_):
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default(), session.as_default():
        K.set_session(session)
        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        saver = tf.train.Saver()

        if FLAGS.testing:
            test(session, graph_ops, saver)
        else:
            train(session, graph_ops, num_actions, saver)  #, saved_path='saved/checkpoints/breakout2.ckpt-2327200')


if __name__ == "__main__":
    tf.app.run()
