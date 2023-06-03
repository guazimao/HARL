"""Critic registry."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.q_critic import QCritic
from harl.algorithms.critics.twin_q_critic import TwinQCritic
from harl.algorithms.critics.soft_twin_q_critic import SoftTwinQCritic
from harl.algorithms.critics.discrete_q_critic import DiscreteQCritic

CRITIC_REGISTRY = {
    "happo": VCritic,
    "hatrpo": VCritic,
    "haa2c": VCritic,
    "mappo": VCritic,
    "haddpg": QCritic,
    "hatd3": TwinQCritic,
    "hasac": SoftTwinQCritic,
    "had3qn": DiscreteQCritic,
    "maddpg": QCritic,
}
