import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential



def mlp_net_boltzmann(input_shape, n_outputs):
    
    inputs = Input(shape = input_shape)

    hidden_layers = [layers.Dense(64, activation = tf.nn.relu, name = "hidden_layers")]
    h_inputs = inputs
    for h_layer in hidden_layers:
        h_inputs = h_layer(h_inputs)
    
    policy_layers = [layers.Dense(256, activation = tf.nn.relu, name = "policy_layers")]
    p_inputs = h_inputs
    for p_layer in policy_layers:
        p_inputs = p_layer(p_inputs)

    value_layers = [layers.Dense(128, activation = tf.nn.relu, name = "value_layers")]
    v_inputs = h_inputs
    for v_layer in value_layers:
        v_inputs = v_layer(v_inputs)
    
    policy_head = layers.Dense(n_outputs, activation = tf.nn.softmax, name = "policy_head")(p_inputs) 
    value_head = layers.Dense(1, activation = tf.nn.tanh, name = "value_head")(v_inputs)
    model = Model(inputs = [inputs], outputs = [policy_head, value_head])

    return model


def mlp_net_gaussian(input_shape, n_outputs):

    inputs = Input(shape = input_shape)

    hidden_layers = [layers.Dense(64, activation = tf.nn.relu, name = "hidden_layers")]
    h_inputs = inputs
    for h_layer in hidden_layers:
        h_inputs = h_layer(h_inputs)
    
    policy_layers = [layers.Dense(128, activation = tf.nn.relu, name = "policy_layers")]
    p_inputs = h_inputs
    for p_layer in policy_layers:
        p_inputs = p_layer(p_inputs)

    value_layers = [layers.Dense(128, activation = tf.nn.relu, name = "value_layers")]
    v_inputs = h_inputs
    for v_layer in value_layers:
        v_inputs = v_layer(v_inputs)
    
    mean = layers.Dense(1, activation = tf.nn.softmax, name = "mean")(p_inputs) 
    std = layers.Dense(1, activation = tf.nn.softmax, name = "std")(p_inputs)
    value_head = layers.Dense(1, activation = tf.nn.tanh, name = "value_head")(v_inputs)
    model = Model(inputs = [inputs], outputs = [mean, std, value_head])

    return model




