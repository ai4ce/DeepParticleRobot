import gym
from gym_dpr.envs.viz import Visualizer
import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot
from gym_dpr.envs.DPR_ParticleRobot import CircularBot
from gym_dpr.envs.DPR_SuperAgent import SuperCircularBot
from gym_dpr.envs.DPR_World import World

import math

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.results_plotter import plot_results, plot_curves

learning_rate = 0.0001
n_steps = 40000
batch_size = 16
n_epochs = 10
gamma = 1.0
gae_lambda = 0.99
clip_range = 0.2
clip_range_vf = None
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 10
use_sde = False
sde_sample_freq = -1
target_kl = None
tensorboard_log = "ppo_object_manipulation_log"
create_eval_env = False
policy_kwargs = dict(net_arch=[64, 128, 128, 64])
verbose = 1
seed = None
device = "auto"
_init_setup_model = True


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

for idx in range(3):
  # REGISTER ENV
  env = gym.make('dpr_single-v0',
                numBots=9, worldClass=World, botClass=CircularBot, superBotClass=SuperCircularBot,
                discreteActionSpace=True, continuousAction=False,
                goalFrame=True,
                rewardFunc="piecewise",
                randomSeed=idx,
                fixedStart=False, fixedGoal=True,
                fixedStartCoords=None, fixedGoalCoords=(0, 0),
                polarStartCoords=True, polarGoalCoords=False,
                xLower=-1000, xUpper=1000, yLower=-1000, yUpper=1000,
                radiusLower=450, radiusUpper=550, angleLower=0, angleUpper=2 * math.pi,
                numDead=0, deadIxs=None,
                gate=False, gateSize=150,
                manipulationTask=True, objectType="Ball", objectPos=None, initializeObjectTangent=True, objectDims=[100, 30],
                visualizer=None, recordInfo=False)
  check_env(env)
  # INITIALIZE MODEL
  model = PPO("MlpPolicy",
              env,
              learning_rate = linear_schedule(0.005),
              n_steps = n_steps,
              batch_size = batch_size,
              n_epochs = n_epochs,
              gamma = gamma,
              gae_lambda = gae_lambda,
              clip_range = clip_range,
              clip_range_vf = clip_range_vf,
              ent_coef = ent_coef,
              vf_coef = vf_coef,
              max_grad_norm = max_grad_norm,
              use_sde = use_sde,
              sde_sample_freq = sde_sample_freq,
              target_kl = target_kl,
              tensorboard_log = tensorboard_log,
              create_eval_env = create_eval_env,
              policy_kwargs = policy_kwargs,
              verbose = verbose,
              seed = seed,
              device = device,
              _init_setup_model = _init_setup_model)

  #TRAIN THE MODEL
  model.learn(total_timesteps=1000000)
  model.save("ppo_dpr_object_manipulation_single_{}".format(idx))

# SAVE AND LOAD
# model.save("ppo_dpr_object_manipulation_single")
# del model
# model = PPO.load("ppo_dpr_object_manipulation_single")
#
# # TEST THE POLICY AND RENDER
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, reward, dones, info = env.step(action)
#     env.render()
#     if dones:
#         break
