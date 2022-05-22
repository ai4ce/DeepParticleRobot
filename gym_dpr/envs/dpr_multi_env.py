import gym
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from statistics import mean

import pymunk
from gym import spaces

import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot
from gym_dpr.envs.DPR_ParticleRobot import CircularBot
from gym_dpr.envs.DPR_World import World


class DPRMultiEnv(gym.Env):
    '''Gym Environment for Decentralized Particle Robot Control'''
    metadata = {'render.modes': ['human']}

    def __init__(self, numBots, worldClass=World, botClass=CircularBot,
                 continuousAction=False,
                 collaborative=True, fullyObservable=True,
                 goalFrame=True,
                 rewardFunc="piecewise",
                 randomSeed=0,
                 fixedStart=False, fixedGoal=True,
                 fixedStartCoords=None, fixedGoalCoords=(0, 0),
                 polarStartCoords=True, polarGoalCoords=False,
                 xLower=-1000, xUpper=1000, yLower=-1000, yUpper=1000,
                 radiusLower=450, radiusUpper=550, angleLower=0, angleUpper=2 * math.pi,
                 numDead=0, deadIxs=None,
                 gate=False, gateSize=150,
                 manipulationTask=False, objectType="Ball", objectPos=None, initializeObjectTangent=True, objectDims=[100, 30],
                 visualizer=None, recordInfo=False):
        random.seed(randomSeed)

        self.num = numBots
        self.dim = math.ceil(math.sqrt(numBots))
        self.world = worldClass()
        self.particleClass = botClass
        self.particles = []

        self.prevCOM = None
        self.currCOM = None

        self.continuousAction = continuousAction
        self.action_space = []
        for i in range(self.num):
            if self.continuousAction:
                space = spaces.Box(low=0, high=math.pi/2, shape=(1,))
            else:
                space = spaces.Discrete(2)
            self.action_space.append(space)

        self.goalFrame = goalFrame
        self.isFullyObservable = fullyObservable
        self.isCollaborative = collaborative
        self.observation_space = []
        for i in range(self.num):
            if fullyObservable:
                if manipulationTask:
                    space = spaces.Box(low=-1, high=1, shape=(4 * numBots + 4,))
                else:
                    space = spaces.Box(low=-1, high=1, shape=(4 * numBots,))
                self.observation_space.append(space)

        if manipulationTask:
            if rewardFunc == "Piecewise":
                self.rewardFunc = self.objectPiecewiseReward
            elif rewardFunc == "Vec reward":
                self.rewardFunc = self.objectVecReward
            elif rewardFunc == "Inverse displacement":
                self.rewardFunc = self.objectPositionReward
            else:
                self.rewardFunc = self.objectPiecewiseReward
        else:
            if collaborative:
                if rewardFunc == "Piecewise":
                    self.rewardFunc = self.cooperativePiecewiseVecReward
                elif rewardFunc == "Vec reward":
                    self.rewardFunc = self.cooperativeVecReward
                elif rewardFunc == "Inverse displacement":
                    self.rewardFunc = self.cooperativePositionReward
                else:
                    self.rewardFunc = self.cooperativePiecewiseVecReward

        self.fixedStart = fixedStart and (fixedStartCoords != None)
        self.fixedGoal = fixedGoal and (fixedGoalCoords != None)
        self.polarStartCoords = polarStartCoords
        self.polarGoalCoords = polarGoalCoords

        if self.fixedStart:
            self.startPos = fixedStartCoords
        else:
            self.startPos = None

        if self.fixedGoal:
            self.goalPos = fixedGoalCoords
        else:
            self.goalPos = None

        self.start = None
        self.goal = None

        self.xLower = xLower
        self.xUpper = xUpper
        self.yLower = yLower
        self.yUpper = yUpper

        self.radiusLower = radiusLower
        self.radiusUpper = radiusUpper
        self.angleLower = angleLower
        self.angleUpper = angleUpper

        self.numDead = numDead
        self.deadIxs = deadIxs

        self.gate = gate
        self.gateSize = gateSize

        self.manipulationTask = manipulationTask
        self.objectType = objectType
        self.objectPos = objectPos
        self.initializeObjectTangent = initializeObjectTangent
        self.objectDims = objectDims

        self.manipulatedObject = None
        self.objectPrevPos = None
        self.objectCurrPos = None

        self.COMtrajectory = []
        self.objectTrajectory = []
        self.rewards = []

        self.visualizer = visualizer
        self.recordInfo = recordInfo
        self.steps = 0
        super(DPRMultiEnv, self).__init__()

    def updateObs(self):
        '''
        Updates the coordinates of the center of mass of the entire particle robot system

        :return:
        '''
        self.prevCOM = self.currCOM
        self.currCOM = self.getParticlesCOM()

    def cooperativePositionReward(self):
        '''
        Basic reward function - dependent on the center of mass

        :return: inverse of the distance between the Center of Mass(COM) of the particle robot(DPR) system and the goal
        '''
        return 1. / np.linalg.norm(self.currCOM - self.goal)

    def cooperativeVecReward(self):
        '''
        goalVec is a vector representing the ideal trajectory (between the COM and the goal)
        actualVec is a vector representing the trajectory of the COM of the DPR system

        :return: length of the projection of actualVec on goalVec
        '''
        goalVec = pymunk.vec2d.Vec2d(self.goal[0] - self.prevCOM[0], self.goal[1] - self.prevCOM[1])
        actualVec = self.currCOM - self.prevCOM
        return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec))

    def cooperativePiecewiseVecReward(self):
        '''
        same as vecReward() except when the COM is within a threshold distance,
        then the reward is multiplied by (threshold distance / distance to goal).

        :return: length of the projection of actualVec on goalVec,
        possibly increased by a factor inversely proportional to the distance between the COM and goal
        '''
        goalVec = pymunk.vec2d.Vec2d(self.goal[0] - self.prevCOM[0], self.goal[1] - self.prevCOM[1])
        actualVec = self.currCOM - self.prevCOM
        dist = np.linalg.norm(self.goal - self.prevCOM)
        if dist < self.radiusLower:
            return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec)) * (self.radiusLower / dist)
        else:
            return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec))

    def updateObjectObs(self):
        '''
        Updates the coordinates of the manipulated object

        :return:
        '''
        self.objectPrevPos = self.objectCurrPos
        self.objectCurrPos = self.manipulatedObject.body.position

    def objectPositionReward(self):
        '''
        Basic manipulation reward function

        :return: inverse of the distance between the manipulated object and the goal
        '''
        return 1. / np.linalg.norm(self.manipulatedObject.body.position - self.goal)

    def objectVecReward(self):
        '''
        goalVec is a vector representing the ideal trajectory (between the object and the goal)
        actualVec is a vector representing the trajectory of the object

        :return: length of the projection of actualVec on goalVec
        '''
        goalVec = pymunk.vec2d.Vec2d(self.goal[0] - self.objectPrevPos[0], self.goal[1] - self.objectPrevPos[1])
        actualVec = self.objectCurrPos - self.objectPrevPos
        return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec))

    def objectPiecewiseReward(self):
        '''
        same as objectVecReward() except when the object is within a threshold distance,
        then the reward is multiplied by (threshold distance / distance to goal).

        :return: length of the projection of actualVec on goalVec,
        possibly increased by a factor inversely proportional to the distance between the object and goal
        '''
        goalVec = pymunk.vec2d.Vec2d(self.goal[0] - self.objectPrevPos[0], self.goal[1] - self.objectPrevPos[1])
        actualVec = self.objectCurrPos - self.objectPrevPos
        dist = np.linalg.norm(self.manipulatedObject.body.position - self.goal)
        if dist < self.radiusLower:
            return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec)) * (self.radiusLower / dist)
        else:
            return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec))

    def getParticleObs(self):
        '''
        Get positions and velocities of individual particles
        bot position transformed to goal reference frame

        :return: np array containing x y components of position and velocity and angle for each particle
        '''
        xs = []
        ys = []
        vxs = []
        vys = []
        for bot in self.particles:
            x, y = bot.shape.body.position
            if self.goalFrame:
                x -= self.goal[0]
                y -= self.goal[1]
            xs.append(x)
            ys.append(y)
            vx, vy = bot.shape.body.velocity
            vxs.append(vx)
            vys.append(vy)
        return np.array([xs, ys, vxs, vys])

    def getParticlesCOM(self):
        '''
        Calculates center of mass of the system of particle robots - assumes every particle robot weighs the same

        :return: vector coordinates of center of mass
        '''
        xs = []
        ys = []
        for bot in self.particles:
            x, y = bot.shape.body.position
            xs.append(x)
            ys.append(y)
        return pymunk.vec2d.Vec2d(mean(xs), mean(ys))

    def observations(self):
        '''
        Scales and flattens observations of particle robots, and in object manipulation task, appends object observations

        :return: scaled and flattened observations (positions and velocities of every agent)
        '''
        def scale(vl, factor=1000.):
            v = np.array(vl)
            return list(v / factor)

        obs = self.getParticleObs()
        if self.manipulationTask:
            obs = np.concatenate([obs, self.world.getObjectObs()])
        obs = scale(obs)
        return np.array(obs)

    def setupWorld(self, startPos, goalPos, deadIxs):
        '''
        Resets particle robot system

        :param startPos: Starting COM of particle robot
        :param goalPos: Goal coordinates
        :param deadIxs: Indices of "dead" particle robots
        :return:
        '''
        def posFunc(centerPos, ix):
            BOT_DIAMETER = 2 * DPR_ParticleRobot.BOT_RADIUS + DPR_ParticleRobot.PADDLE_WIDTH + DPR_ParticleRobot.PADDLE_LENGTH
            xc, yc = centerPos
            x = ((ix % self.dim) - (self.dim / 2) + 0.5) * BOT_DIAMETER + xc
            y = ((ix // self.dim) - (self.dim / 2) + 0.5) * BOT_DIAMETER + yc
            return (x, y)

        for i in range(self.num):
            dead = i in deadIxs
            bot = self.particleClass(pos=posFunc(startPos, i), botId=i, dead=dead)
            self.particles.append(bot)
            self.world.addParticleRobot(bot)

        self.start = startPos
        self.goal = goalPos

    def polar2rect(self, radius, angle):
        '''
        Converts polar coordinates to cartesian coordinates

        :param radius: radius/magnitude
        :param angle: angle
        :return: Cartesian coordinates
        '''
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return (x, y)

    def initializeObject(self, start, goal):
        '''
        Function to initialize target object position. Places the object adjacent to the Particle Robot system

        :param start: Particle Robot COM start position
        :param goal: Object goal location
        :return: Object start position
        '''
        BOT_DIAMETER = 2 * DPR_ParticleRobot.BOT_RADIUS + DPR_ParticleRobot.PADDLE_WIDTH + DPR_ParticleRobot.PADDLE_LENGTH
        radius = (self.dim / 2) * BOT_DIAMETER + (self.objectDims[1] / 2)
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        l = max(abs(dx), abs(dy))
        s = min(abs(dx), abs(dy))
        side = radius * s / l
        if abs(dx) > abs(dy):
            x = start[0] + (radius * (dx / abs(dx)))
            y = start[1] + (side * (dy / abs(dy)))
        else:
            x = start[0] + (side * (dx / abs(dx)))
            y = start[1] + (radius * (dy / abs(dy)))
        return x, y

    def reset(self, startPos=None, goalPos=None,
              deadIxs=None,
              objectType=None, objectPos=None, objectDims=None,
              gate=False):
        '''
        Resets and initializes world

        :param startPos: start position
        :param goalPos: goal position
        :param deadIxs: indices of "dead" particle robots
        :param objectType: target object type (ball/box)
        :param objectPos: target object start position
        :param objectDims: List containing [object size, object mass]
        :param gate: initialize gate?
        :return: Inital world observation
        '''
        self.world.removeAll()

        if startPos == None:
            if self.polarStartCoords:
                if self.startPos != None:
                    start = self.polar2rect(self.startPos)
                else:
                    r = random.randrange(self.radiusLower, self.radiusUpper)
                    phi = random.uniform(self.angleLower, self.angleUpper)
                    start = self.polar2rect(r, phi)
            else:
                if self.startPos != None:
                    start = self.startPos
                else:
                    start = (random.randrange(self.xLower, self.xUpper), random.randrange(self.xLower, self.xUpper))
        else:
            if self.polarStartCoords:
                start = self.polar2rect(startPos)
            else:
                start = startPos

        if goalPos == None:
            if self.polarGoalCoords:
                if self.goalPos != None:
                    goal = self.polar2rect(self.goalPos)
                else:
                    r = random.randrange(self.radiusLower, self.radiusUpper)
                    phi = random.uniform(self.angleLower, self.angleUpper)
                    goal = self.polar2rect(r, phi)
            else:
                if self.goalPos != None:
                    goal = self.goalPos
                else:
                    goal = (random.randrange(self.xLower, self.xUpper), random.randrange(self.xLower, self.xUpper))
        else:
            if self.polarGoalCoords:
                goal = self.polar2rect(goalPos)
            else:
                goal = goalPos

        if deadIxs != None:
            dead = deadIxs
        else:
            if self.deadIxs != None:
                dead = self.deadIxs
            else:
                dead = list(np.random.choice(self.num, size=self.numDead, replace=False))

        if gate:
            self.world.addGate(goal, start, self.gateSize)
        elif self.gate:
            self.world.addGate(goal, start, self.gateSize)
        else:
            pass

        if self.manipulationTask:
            if objectType != None:
                objType = objectType
            else:
                objType = self.objectType

            if objectPos != None:
                objPos = objectPos
            if self.initializeObjectTangent:
                objPos = self.initializeObject(start, goal)
            else:
                objPos = self.objectPos

            if objectDims != None:
                objParams = objectDims
            else:
                objParams = self.objectDims

            objArgs = [objPos]
            for arg in objParams:
                objArgs.append(arg)

            self.manipulatedObject = self.world.addObject(objType, objArgs)
            self.objectPrevPos = objPos
            self.objectCurrPos = objPos

        self.setupWorld(start, goal, dead)
        self.steps = 0

        return self.observations()

    def render(self, mode='human'):
        '''
        Uses Pygame for visualization

        :param mode: Just human for now
        :return:
        '''
        if self.visualizer != None:
            self.visualizer.viz(timestep=self.steps, world=self.world)

    def step(self, actions_list):
        '''
        Simulates one step of the environment
        1. gets observations
        2. simulates magnetic attraction
        3. takes actions
        4. run timesteps in pymunk
        5. reset magnets
        6. calculate reward
        7. determine state of environment
        8. record information

        :param actions_list: list of actions for each particle robot
        :return: world observations, scalar reward, environment state, information
        '''
        self.steps += 1

        # 1. observations
        self.updateObs()
        if self.manipulationTask:
            self.updateObjectObs()
        observations = self.observations()

        # 2. react/create magnets
        self.world.botReact()
        # self.world.addMagnets()

        # 3. take actions
        for ix, action in enumerate(actions_list):
            self.particles[ix].act(action)

        # 4. simulate
        for i in range(self.world.pymunk_steps_per_frame):
            self.world.space.step(self.world._dt)

        # 5. clear current magnets
        self.world.botRemoveMagnets()
        # self.world.removeMagnets()

        # 6. rewards
        rewards = self.rewardFunc()

        # 7. check system state (over 2,500 steps, or center of mass is less than 10 units from goal)
        if self.manipulationTask == True:
            done = bool((self.steps > 2500) or (np.linalg.norm(self.manipulatedObject.body.position - self.goal) < 10))
        else:
            done = bool((self.steps > 2500) or (np.linalg.norm(self.agent.getCOM() - self.goal) < 10))

        #  8. system information
        if self.recordInfo:
            self.COMtrajectory.append(self.currCOM)
            if self.manipulationTask:
                self.objectTrajectory.append(self.manipulatedObject.body.position)
            self.rewards.append(rewards)
            info = {"COM trajectory": self.COMtrajectory, "Object trajectory": self.objectTrajectory, "Reward": self.rewards}
        else:
            info = {}

        return observations, rewards, done, info
