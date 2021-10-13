# DeepParticleRobot

[**Wenyu Han**](https://github.com/WenyuHan-LiNa), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng), [**Haoran Wu**](https://www.linkedin.com/in/haoran-lucas-ng-4053471a0/), [**Alexander Gao**](https://www.alexandergao.com/), [**Armand Jordana**](https://wp.nyu.edu/machinesinmotion/people/), [**Dong Liu**](http://mechatronics.engineering.nyu.edu/people/phd-candidates/dongdong-liu.php), [**Lerrel Pinto**](https://www.lerrelpinto.com/), [**Ludovic Righetti**](https://wp.nyu.edu/machinesinmotion/89-2/)

## Abstract
Particle  robots  are  novel  biologically-inspiredrobotic systems where locomotion can be achieved collectivelyand robustly, but not independently. While its control iscurrently limited to a hand-crafted policy for basic locomotiontasks, such a multi-robot system can be potentially controlledvia Deep Reinforcement Learning (DRL) for different tasks.However, the particle robot system presents a new set ofchallenges for DRL differing from existing swarm roboticssystems: the low dimensionality of each robot and the increasednecessity of coordination between robots. We present a 2Dparticle robot simulator using the OpenAI Gym interface andPymunk as the physics engine, and introduce new tasks andchallenges to research the underexplored applications of DRLin the particle robot system. Moreover, we use Stable-baselines3to provide a set of benchmarks for the tasks. Current baselineDRL algorithms show signs of achieving the tasks but areunable to reach the performance of the hand-crafted policy.Further development of DRL algorithms is necessary in orderto accomplish the proposed tasks.

## [Code (GitHub)](https://github.com/ai4ce/SNAC) & Dependencies
All environment scripts can be found in [Env](https://github.com/ai4ce/SNAC/tree/main/Env) folder. These environments are developed based on the [OpenAi Gym](https://gym.openai.com/). All baseline scripts are in [script](https://github.com/ai4ce/SNAC/tree/main/script) floder. You need to install the [Pytorch](https://pytorch.org/) to run all baseline scripts. We use [Stable baseline](https://github.com/openai/baselines/) for PPO algorithm. 
## How to use

Our environment is developed based on the [OpenAi Gym](https://gym.openai.com/). You can simply follow the similar way to use our environment. Here we present an example for using 1D static task environment.
```
from DMP_Env_1D_static import deep_mobile_printing_1d1r ### you may need to find the path to this environment in [Env] folder 
env = deep_mobile_printing_1d1r(plan_choose=2) ### plan_choose could be 0: sin, 1: Gaussian, and 2: Step curve  
observation = env.reset()
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.clear()
for _ in range(1000):
  action = np.random.randint(env.action_dim) # your agent here (this takes random actions)
  observation, reward, done = env.step(action)
  env.render(ax)
  plt.pause(0.1)
  if done:
    break
plt.show()
```

## [Paper (arXiv)](https://arxiv.org/abs/2103.16732)
To cite our paper:
```
@misc{han2021simultaneous,
      title={Simultaneous Navigation and Construction Benchmarking Environments}, 
      author={Wenyu Han and Chen Feng and Haoran Wu and Alexander Gao and Armand Jordana and Dong Liu and Lerrel Pinto and Ludovic Righetti},
      year={2021},
      eprint={2103.16732},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Acknowledgment
 The research is supported by NSF CPS program under CMMI-1932187. The authors gratefully thank our human test participants and the helpful comments from [**Bolei Zhou**](http://bzhou.ie.cuhk.edu.hk/), [**Zhen Liu**](http://itszhen.com/), and the anonymous reviewers, and also [**Congcong Wen**](https://scholar.google.com/citations?hl=en&user=OTBgvCYAAAAJ) for paper revision.
