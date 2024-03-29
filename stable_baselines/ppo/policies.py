import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential

from stable_baselines.common.policies import BasePolicy, register_policy, MlpExtractor
from stable_baselines.common.distributions import make_proba_distribution, DiagGaussianDistribution, CategoricalDistribution


class PPOPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for A2C and derivates (PPO).

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param learning_rate: (callable) Learning rate schedule (could be constant)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param activation_fn: (nn.Module) Activation function
    :param adam_epsilon: (float) Small values to avoid NaN in ADAM optimizer
    :param ortho_init: (bool) Whether to use or not orthogonal initialization
    :param log_std_init: (float) Initial value for the log standard deviation
    """
    def __init__(self, observation_space, action_space, learning_rate, net_arch=None, activation_fn=tf.nn.tanh, adam_epsilon=1e-5, ortho_init=True, log_std_init=0.0):

        super(PPOPolicy, self).__init__(observation_space, action_space)
        self.obs_dim = self.observation_space.shape[0]

        # Default network architecture, from stable-baselines

        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.adam_epsilon = adam_epsilon
        self.ortho_init = ortho_init
        self.net_args = {
            'input_dim': self.obs_dim,
            'output_dim': -1,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn
        }
        self.shared_net = None
        self.pi_net, self.vf_net = None, None

        # In the future, feature_extractor will be replaced with a CNN

        self.features_extractor = Sequential(layers.Flatten(input_shape=(self.obs_dim,), dtype=tf.float32))
        self.features_dim = self.obs_dim
        self.log_std_init = log_std_init
        dist_kwargs = None

        # Action distribution

        self.action_dist = make_proba_distribution(action_space, dist_kwargs=dist_kwargs)

        self._build(learning_rate)

    def _build(self, learning_rate):
        self.mlp_extractor = MlpExtractor(self.obs_dim, self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi, log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        self.value_net = Sequential(layers.Dense(1, input_shape=(self.mlp_extractor.latent_dim_vf,)))

        self.features_extractor.build()
        self.action_net.build()
        self.value_net.build()

        self.build(input_shape=(None, self.obs_dim))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate(1), epsilon=self.adam_epsilon)

    def save(self, path):
        self.save_weights(path)

    def load(self, path):
        self.load_weights(path)

    @tf.function
    def call(self, obs, deterministic=False):
        latent_pi, latent_vf = self._get_latent(obs)
        value = self.value_net(latent_vf)
        action_logits, action, action_distribution = self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)
        log_prob = action_distribution.log_prob(action)
        return action, value, log_prob, action_logits

    def _get_latent(self, obs):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        return latent_pi, latent_vf

    def _get_action_dist_from_latent(self, latent_pi, deterministic=False):
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return mean_actions, *self.action_dist.proba_distribution(mean_actions, self.log_std, deterministic=deterministic)

        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return mean_actions, *self.action_dist.proba_distribution(mean_actions, deterministic=deterministic)

    def actor_forward(self, obs, deterministic=False):
        latent_pi, _ = self._get_latent(obs)
        _, action, _ = self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)
        return tf.stop_gradient(action).numpy()

    @tf.function
    def evaluate_actions(self, obs, action, deterministic=False):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param action: (th.Tensor)
        :param deterministic: (bool)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf = self._get_latent(obs)
        _, _, action_distribution = self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)
        log_prob = action_distribution.log_prob(action)
        value = self.value_net(latent_vf)
        return value, log_prob, action_distribution.entropy()

    def value_forward(self, obs):
        _, latent_vf, _ = self._get_latent(obs)
        return self.value_net(latent_vf)

MlpPolicy = PPOPolicy

register_policy("MlpPolicy", MlpPolicy)
