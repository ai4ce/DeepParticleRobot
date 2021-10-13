from gym.envs.registration import register

register(
    id='dpr_single-v0',
    entry_point='gym_dpr.envs.dpr_single_env:DPRSingleEnv',
)

register(
    id='dpr_multi-v0',
    entry_point='gym_dpr.envs.dpr_multi_env:DPRMultiEnv',
)