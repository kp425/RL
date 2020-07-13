import tensorflow as tf
import tensorflow_probability as tfp
import gym

from RL.networks.Networks import *

class CSerializable:

    def __init__(self, path):
        self.__path = path

    def _save(self, model):
        if self.__path is not None:
            tf.keras.models.save_model(model, self.__path)

    def load(self):
        return tf.keras.models.load_model(self.__path)
    
    def get_path(self):
        return self.__path


class Policy(CSerializable):
    def __init__(self, input_shape, n_outputs, net = None, model_path=None):
        
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.model_path = model_path
        super(Policy, self).__init__(self.model_path)
        self._net = self._get_net(net = net, model_path = model_path)
        self.trainable_variables = self._net.trainable_variables
    
    def _get_net(self, net = None, model_path= None):

        if net == None and model_path == None:
            raise "Function approximator isn't passed"
        network = None
        if model_path is None:
            network = net(self.input_shape, self.n_outputs)
        else:
            try:
                network = self.load()
                print("Loaded from path")
            except OSError:
                print("Using default net...")
                network = net(self.input_shape, self.n_outputs)
        return network
        
        
    #Override this
    def __call__(self, state):
        pass
    
    def get_net(self):
        return self._net
    
    def get_architecture(self):
        self._net.summary()
        return tf.keras.utils.plot_model(self._net, "net.png", show_shapes=True)
    
    def save(self):
        if self.model_path is not None:
            self._save(self._net)


class BoltzmannPolicy(Policy):
    def __init__(self, state_spec, action_spec, net = None, model_path=None):

        self.state_spec = state_spec
        self.action_spec = action_spec
        super(BoltzmannPolicy, self).__init__(self.state_spec.shape, self.action_spec.n, 
                                              net = net, model_path= model_path)

    def __call__(self, state):
        if state.shape == self.state_spec.shape:
            state = tf.reshape(state, shape = [-1,*state.shape])
        probs, value = self._net(state)
        dist = tfp.distributions.Categorical(probs = probs)
        return dist.sample(), dist, value

class GaussianPolicy(Policy):
    def __init__(self, state_spec, action_spec, net = None, model_path=None):
        self.state_spec = state_spec
        self.action_spec = action_spec
        super(GaussianPolicy, self).__init__(state_spec.shape, 1, 
                                            net = net, model_path= model_path)

    def __call__(self, state):
        if state.shape == self.input_shape:
            state = tf.reshape(state, shape = [-1,*state.shape])
        mean, std, value = self._net(state)
        dist = tfp.distributions.Normal(mean, std)
        action = tf.clip_by_value(dist.sample(), self.action_spec.low, self.action_spec.high)
        return action, dist, value


def make_policy(state_spec, action_spec, net = None, save_path = None):

    if isinstance(action_spec, gym.spaces.Discrete):
        
        if net == None: net = mlp_net_boltzmann
        return BoltzmannPolicy(state_spec, action_spec, net = net,
                               model_path = save_path)

    elif isinstance(action_spec, gym.spaces.Box):
        if net == None: net = mlp_net_gaussian
        return GaussianPolicy(state_spec, action_spec, net = net,
                             model_path = save_path)


