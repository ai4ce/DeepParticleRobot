import gym
from gym_dpr.envs.viz import Visualizer
import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot
from gym_dpr.envs.DPR_ParticleRobot import CircularBot
from gym_dpr.envs.DPR_SuperAgent import SuperCircularBot
from gym_dpr.envs.DPR_World import World

import math

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.results_plotter import plot_results, plot_curves

learning_rate=0.001 # using linear schedule
buffer_size=240000
learning_starts=0
batch_size=16
tau=1.0
gamma=1.0
train_freq=4
gradient_steps=1
replay_buffer_class=None
replay_buffer_kwargs=None
optimize_memory_usage=False
target_update_interval=40000
exploration_fraction=0.1
exploration_initial_eps=1.0
exploration_final_eps=0.065
max_grad_norm=10
tensorboard_log="dqn_simple_navigation_log"
create_eval_env=False
policy_kwargs=dict(net_arch=[64, 128, 128, 128, 64])
verbose=1
seed=None
device='auto'
_init_setup_model=True


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
                 manipulationTask=False, objectType="Ball", objectPos=None, initializeObjectTangent=True, objectDims=[100, 30],
                 visualizer=None, recordInfo=False)
  check_env(env)

  # INITIALIZE MODEL
  model = DQN("MlpPolicy",
              env,
              learning_rate = linear_schedule(0.005),
              buffer_size=buffer_size,
              learning_starts=learning_starts,
              batch_size=batch_size,
              tau=tau,
              gamma=gamma,
              train_freq=train_freq,
              gradient_steps=gradient_steps,
              replay_buffer_class=replay_buffer_class,
              replay_buffer_kwargs=replay_buffer_kwargs,
              optimize_memory_usage=optimize_memory_usage,
              target_update_interval=target_update_interval,
              exploration_fraction=exploration_fraction,
              exploration_initial_eps=exploration_initial_eps,
              exploration_final_eps=exploration_final_eps,
              max_grad_norm=max_grad_norm,
              tensorboard_log=tensorboard_log,
              create_eval_env=create_eval_env,
              policy_kwargs=policy_kwargs,
              verbose=verbose,
              seed=seed,
              device=device,
              _init_setup_model=_init_setup_model)

  # TRAIN THE MODEL
  model.learn(total_timesteps=1000000)
  model.save("deepq_dpr_simple_navigation_single_{}".format(idx))

# SAVE AND LOAD
# model.save("deepq_dpr_simple_navigation_single")
# del model
# model = DQN.load("deepq_dpr_simple_navigation_single")
#
# # TEST THE POLICY AND RENDER
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, reward, dones, info = env.step(action)
#     env.render(2500)
#     if dones:
#         break
