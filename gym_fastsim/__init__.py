import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FastsimSimpleNavigation-v0',
    entry_point='gym_fastsim.simple_nav:SimpleNavEnv',
)
