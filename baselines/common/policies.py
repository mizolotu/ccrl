import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype

import gym


class PolicyWithValue(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        print(policy_network.summary())
        self.policy_network = policy_network
        self.value_network = value_network or policy_network
        self.estimate_q = estimate_q

        # Based on the action space, will select what probability distribution type

        if type(policy_network.output_shape)==list and len(policy_network.output_shape) == 3:
            self.pdtype = make_pdtype(policy_network.output_shape[0], ac_space, init_scale=0.01)
            self.initial_state = policy_network.input_shape[1][1]
        else:
            self.pdtype = make_pdtype(policy_network.output_shape, ac_space, init_scale=0.01)
            self.initial_state = None

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(self.value_network.output_shape, 'q', ac_space.n)
        else:
            if type(policy_network.output_shape) == list and len(policy_network.output_shape) == 3:
                self.value_fc = fc(self.value_network.output_shape[0], 'vf', 1)
            else:
                self.value_fc = fc(self.value_network.output_shape, 'vf', 1)

    @tf.function
    def step(self, observation, state):

        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        if state is not None:
            latent, new_state_h, new_state_c = self.policy_network([observation, state[0], state[1]])
            value_latent, _, _ = self.value_network([observation, state[0], state[1]])
            new_state = [new_state_h, new_state_c]
        else:
            latent = self.policy_network(observation)
            value_latent = self.value_network(observation)
            new_state = None
        pd, pi = self.pdtype.pdfromlatent(latent)
        action = pd.sample()
        neglogp = pd.neglogp(action)
        vf = tf.squeeze(self.value_fc(value_latent), axis=1)
        return action, vf, new_state, neglogp

    @tf.function
    def value(self, observation, state=None):

        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """

        if state is not None and len(state) == 2 and state[0] is not None and state[1] is not None:
            value_latent, _, _ = self.value_network([observation, state[0], state[1]])
        else:
            value_latent = self.value_network(observation)
        result = tf.squeeze(self.value_fc(value_latent), axis=1)
        return result

