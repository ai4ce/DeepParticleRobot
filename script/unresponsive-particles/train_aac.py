import gym
from gym_dpr.envs.viz import Visualizer
import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot
from gym_dpr.envs.DPR_ParticleRobot import CircularBot
from gym_dpr.envs.DPR_SuperAgent import SuperCircularBot
from gym_dpr.envs.DPR_World import World

import math

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.results_plotter import plot_results, plot_curves

learning_rate = 0.00075
n_steps = 40000
gamma = 1.0
gae_lambda = 0.99
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 10
rms_prop_eps = 1e-5
use_rms_prop = True
use_sde = False
sde_sample_freq = -1
normalize_advantage = False
tensorboard_log = "a2c_unresponsive_particles_log"
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

for idx in range(idx):
  # REGISTER ENV
  env = gym.make('dpr_single-v0',
                numBots=9, worldClass=World, botClass=CircularBot, superBotClass=SuperCircularBot,
                discreteActionSpace=False, continuousAction=False,
                goalFrame=True,
                rewardFunc="piecewise",
                randomSeed=idx,
                fixedStart=False, fixedGoal=True,
                fixedStartCoords=None, fixedGoalCoords=(0, 0),
                polarStartCoords=True, polarGoalCoords=False,
                xLower=-1000, xUpper=1000, yLower=-1000, yUpper=1000,
                radiusLower=450, radiusUpper=550, angleLower=0, angleUpper=2 * math.pi,
                numDead=2, deadIxs=None,
                gate=False, gateSize=150,
                manipulationTask=False, objectType="Ball", objectPos=None, initializeObjectTangent=True, objectDims=[100, 30],
                visualizer=None, recordInfo=False)
  check_env(env)

  # INITIALIZE MODEL
  model = A2C("MlpPolicy",
              env,
              learning_rate = linear_schedule(0.005),
              n_steps = n_steps,
              gamma = gamma,
              gae_lambda = gae_lambda,
              ent_coef = ent_coef,
              vf_coef = vf_coef,
              max_grad_norm = max_grad_norm,
              rms_prop_eps = rms_prop_eps,
              use_rms_prop = use_rms_prop,
              use_sde = use_sde,
              sde_sample_freq = sde_sample_freq,
              normalize_advantage = normalize_advantage,
              tensorboard_log = tensorboard_log,
              create_eval_env = create_eval_env,
              policy_kwargs = policy_kwargs,
              verbose = verbose,
              seed = seed,
              device = device,
              _init_setup_model = _init_setup_model)

  # TRAIN THE MODEL
  model.learn(total_timesteps=1000000)
  model.save("a2c_dpr_unresponsive_particles_single_{}".format(idx))

# SAVE AND LOAD
# model.save("a2c_dpr_unresponsive_particles_single")
# del model
# model = A2C.load("a2c_dpr_unresponsive_particles_single")
#
# # TEST THE POLICY AND RENDER
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, reward, dones, info = env.step(action)
#     env.render(2500)
#     if dones:
#         break
