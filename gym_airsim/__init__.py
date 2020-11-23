from gym.envs.registration import register

register(
    id='airsim-v0',
    entry_point='gym_airsim.envs:AirSimEnv',
    max_episode_steps=501,
)
