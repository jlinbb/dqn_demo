import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model


def build_network(num_actions, agent_history_length, width, height, name_scope):
    with tf.device("/cpu:0"):
        with tf.name_scope(name_scope):
            state = tf.placeholder(tf.float32, [None, agent_history_length, width, height], name="state")
            inputs = Input(shape=(agent_history_length, width, height,))
            model = Conv2D(16, (8,8), strides=(4,4), activation='relu', padding='same', data_format='channels_first')(inputs)
            model = Conv2D(32, (4,4), strides=(2,2), activation='relu', padding='same', data_format='channels_first')(model)
            model = Flatten()(model)
            model = Dense(256, activation='relu')(model)
            q_values = Dense(num_actions)(model)
            m = Model(inputs=inputs, outputs=q_values)
    return state, m