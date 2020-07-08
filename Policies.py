import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential
import tensorflow_probability as tfp



def build_net(input_shape, n_outputs):

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


class CSerializable:

    def __init__(self, path):
        self.__path = path

    def _save(self, model):
        tf.keras.models.save_model(model, self.__path)

    def load(self):
        return tf.keras.models.load_model(self.__path)
    
    def get_path(self):
        return self.__path
        

class BoltzmannPolicy(CSerializable):
    def __init__(self, input_shape, n_outputs, net = None, model_path=None):
        
        self.model_path = model_path
        super(BoltzmannPolicy, self).__init__(self.model_path)
        self.input_shape = input_shape
        self.n_outputs = n_outputs

        if net is not None:
            self.__net = net
        else:
            if model_path is not None:
                self.__net = self.load()
                try:
                    self.__net = self.load()
                    print("Loaded from path")
                except OSError:
                    print("Using default net...")
                    self.__net = build_net(self.input_shape, self.n_outputs)
            else:
                print("Using default net...")
                self.__net = build_net(self.input_shape, self.n_outputs)
        
        self.trainable_variables = self.__net.trainable_variables
        

    def __call__(self, state):
        if state.shape == self.input_shape:
            state = tf.reshape(state, shape = [-1,*state.shape])
        probs, value = self.__net(state)
        dist = tfp.distributions.Categorical(probs = probs)
        return dist.sample(), dist, value
    
    def get_net(self):
        return self.__net
    
    def get_architecture(self):
        self.__net.summary()
        return tf.keras.utils.plot_model(self.__net, "net.png", show_shapes=True)
    
    def save(self):
        self._save(self.__net)

