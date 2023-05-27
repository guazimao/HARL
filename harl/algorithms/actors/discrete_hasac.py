import torch
from harl.utils.envs_tools import check
from harl.models.policy_models.discrete_policy import DiscretePolicy
from harl.utils.discrete_util import gumbel_softmax
from harl.algorithms.actors.off_policy_base import OffPolicyBase


class DiscreteHASAC(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert act_space.__class__.__name__ == 'Discrete' or act_space.__class__.__name__ == 'MultiDiscrete', "only discrete action space is supported by discrete_hasac."
        """Initialize DiscreteHASAC algorithm."""
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self._multidiscrete_action = False
        if act_space.__class__.__name__ == 'MultiDiscrete':
            self._multidiscrete_action = True
        # create actor network
        self.actor = DiscretePolicy(args, obs_space, act_space, device)
        # create actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr)
        self.turn_off_grad()

    def get_actions(self, obs, available_actions=None, stochastic=True):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions, _ = self.actor(obs, available_actions, stochastic)
        return actions

    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        """
        get actions and logprobs of actions for observations.
        params:
        obs: (threads, dim)
        available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
        stochastic: (bool) stochastic actions or deterministic actions
        returns:
        actions: (threads, dim)
        """
        obs = check(obs).to(**self.tpdv)
        if self._multidiscrete_action:
            probs = self.actor.get_probs(obs, available_actions)
            actions = []
            logp_actions = []
            for prob in probs:
                logits = torch.log(prob + 1e-8)
                action = gumbel_softmax(logits, hard=True, device=self.device)
                logp_action = torch.sum(action * logits, dim=-1, keepdim=True)
                actions.append(action)
                logp_actions.append(logp_action)
            actions = torch.cat(actions, dim=-1)
            logp_actions = torch.cat(logp_actions, dim=-1)
        else:
            logits = torch.log(self.actor.get_probs(obs, available_actions) + 1e-8)
            actions = gumbel_softmax(logits, hard=True, device=self.device)
            logp_actions = torch.sum(actions * logits, dim=-1, keepdim=True)
        return actions, logp_actions

    def save(self, save_dir, id):
        """Save the actor."""
        torch.save(self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt")

    def restore(self, model_dir, id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)
