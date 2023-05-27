import torch
import torch.nn as nn
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.act import ACTLayer



class DiscretePolicy(nn.Module):
    """
    Actor network class for DiscreteHASAC. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(DiscretePolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args
        )

        self.to(device)

    def forward(self, obs, available_actions=None, stochastic=True):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param stochastic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs

    def get_probs(self, obs, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        return self.act.get_probs(actor_features, available_actions)