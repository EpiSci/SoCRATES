from gym.envs.registration import register

register(
    id='Dsgym3-v0',
    entry_point='gym_ds3.envs.core.ds3_env:DS3GymEnv',
)
