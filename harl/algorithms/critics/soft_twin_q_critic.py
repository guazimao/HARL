"""Soft Twin Q Critic."""
import numpy as np
import torch
from copy import deepcopy
from harl.models.value_function_models.continuous_q_net import ContinuousQNet
from harl.models.value_function_models.discrete_q_net import DiscreteQNet
import torch.nn.functional as F
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule
import itertools


class SoftTwinQCritic:
    """Soft Twin Q Critic.
    Critic that learns two soft Q-functions. The action space can be continuous and discrete.
    """

    def __init__(self, args, share_obs_space, act_space, num_agents, state_type, device=torch.device("cpu")):
        """Initialize the critic."""
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = act_space[0].__class__.__name__
        if act_space[0].__class__.__name__ == "Box":
            self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
            self.critic2 = ContinuousQNet(args, share_obs_space, act_space, device)
        else:
            self.critic = DiscreteQNet(args, share_obs_space, act_space, device)
            self.critic2 = DiscreteQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False
        self.auto_alpha = args['auto_alpha']
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args['alpha_lr'])
            self.alpha = torch.exp(self.log_alpha.detach())
        else:
            self.alpha = args['alpha']
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]
        critic_params = itertools.chain(self.critic.parameters(), self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr,
        )
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.critic_optimizer, step, steps, self.critic_lr)

    def soft_update(self):
        """Soft update the target networks."""
        for param_target, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
        for param_target, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def update_alpha(self, logp_actions, target_entropy):
        """Auto-tune the temperature parameter alpha."""
        log_prob = torch.sum(
            torch.cat(logp_actions, dim=-1), dim=-1, keepdim=True
        ).detach().to(**self.tpdv) + target_entropy
        alpha_loss = -(self.log_alpha * log_prob).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha.detach())

    def get_values(self, share_obs, actions):
        """Get the soft Q values for the given observations and actions."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return torch.min(self.critic(share_obs, actions), self.critic2(share_obs, actions))

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        valid_transition,
        term,
        next_share_obs,
        next_actions,
        next_logp_actions,
        gamma,
        value_normalizer=None
    ):
        """Train the critic.
        Args:
            share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            actions: (n_agents, batch_size, dim)
            reward: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            done: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            valid_transition: (n_agents, batch_size, 1)
            term: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            next_share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            next_actions: (n_agents, batch_size, dim)
            next_logp_actions: (n_agents, batch_size, 1)
            gamma: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            value_normalizer: (PopArt) normalize the rewards, denormalize critic outputs.
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"

        share_obs = check(share_obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions = check(actions).to(**self.tpdv)
            actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        else:
            actions = check(actions).to(**self.tpdv_a)
            one_hot_actions = []
            for agent_id in range(len(actions)):
                if self.action_type == 'MultiDiscrete':
                    action_dims = self.act_space[agent_id].nvec
                    one_hot_action = []
                    for dim in range(len(action_dims)):
                        one_hot = F.one_hot(actions[agent_id, :, dim], num_classes=action_dims[dim])
                        one_hot_action.append(one_hot)
                    one_hot_action = torch.cat(one_hot_action, dim=-1)
                else:
                    one_hot_action = F.one_hot(actions[agent_id], num_classes=self.act_space[agent_id].n)
                one_hot_actions.append(one_hot_action)
            actions = torch.squeeze(torch.cat(one_hot_actions, dim=-1), dim=1).to(**self.tpdv_a)
        if self.state_type == "FP":
            actions = torch.tile(actions, (self.num_agents, 1))
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        valid_transition = check(np.concatenate(valid_transition, axis=0)).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        if self.action_type == "Box":
            next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        else:
            next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv_a)
        next_logp_actions = torch.sum(torch.cat(next_logp_actions, dim=-1), dim=-1, keepdim=True).to(**self.tpdv)
        if self.state_type == "FP":
            next_actions = torch.tile(next_actions, (self.num_agents, 1))
            next_logp_actions = torch.tile(next_logp_actions, (self.num_agents, 1))
        next_q_values1 = self.target_critic(next_share_obs, next_actions)
        next_q_values2 = self.target_critic2(next_share_obs, next_actions)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        if self.use_proper_time_limits:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                        check(value_normalizer.denormalize(next_q_values)).to(**self.tpdv) - self.alpha * next_logp_actions
                ) * (1 - term)
                q_targets = check(value_normalizer(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (next_q_values - self.alpha * next_logp_actions) * (1 - term)
        else:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                        check(value_normalizer.denormalize(next_q_values)).to(**self.tpdv) - self.alpha * next_logp_actions
                ) * (1 - done)
                q_targets = value_normalizer(q_targets).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (next_q_values - self.alpha * next_logp_actions) * (1 - done)
        if self.use_huber_loss:
            if self.state_type == "FP" and self.use_policy_active_masks:
                critic_loss1 = torch.sum(
                    F.huber_loss(self.critic(share_obs, actions), q_targets, delta=self.huber_delta) * valid_transition
                ) / valid_transition.sum()
                critic_loss2 = torch.mean(
                    F.huber_loss(self.critic2(share_obs, actions), q_targets, delta=self.huber_delta) * valid_transition
                ) / valid_transition.sum()
            else:
                critic_loss1 = torch.mean(
                    F.huber_loss(self.critic(share_obs, actions), q_targets, delta=self.huber_delta)
                )
                critic_loss2 = torch.mean(
                    F.huber_loss(self.critic2(share_obs, actions), q_targets, delta=self.huber_delta)
                )
        else:
            if self.state_type == "FP" and self.use_policy_active_masks:
                critic_loss1 = torch.sum(
                    F.mse_loss(self.critic(share_obs, actions), q_targets) * valid_transition
                ) / valid_transition.sum()
                critic_loss2 = torch.sum(
                    F.mse_loss(self.critic2(share_obs, actions), q_targets) * valid_transition
                ) / valid_transition.sum()
            else:
                critic_loss1 = torch.mean(
                    F.mse_loss(self.critic(share_obs, actions), q_targets)
                )
                critic_loss2 = torch.mean(
                    F.mse_loss(self.critic2(share_obs, actions), q_targets)
                )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, save_dir):
        """Save the model parameters."""
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt"
        )
        torch.save(self.critic2.state_dict(), str(save_dir) + "/critic_agent2" + ".pt")
        torch.save(
            self.target_critic2.state_dict(),
            str(save_dir) + "/target_critic_agent2" + ".pt"
        )

    def restore(self, model_dir):
        """Restore the model parameters."""
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(str(model_dir) + "/target_critic_agent" + ".pt")
        self.target_critic.load_state_dict(target_critic_state_dict)
        critic_state_dict2 = torch.load(str(model_dir) + "/critic_agent2" + ".pt")
        self.critic2.load_state_dict(critic_state_dict2)
        target_critic_state_dict2 = torch.load(str(model_dir) + "/target_critic_agent2" + ".pt")
        self.target_critic2.load_state_dict(target_critic_state_dict2)

    def turn_on_grad(self):
        """Turn on the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
