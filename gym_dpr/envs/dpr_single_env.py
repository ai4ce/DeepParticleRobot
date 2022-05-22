import gym
import matplotlib.pyplot as plt
import numpy as np
import math
import random

import pymunk
from gym import spaces

import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot
from gym_dpr.envs.DPR_ParticleRobot import CircularBot
from gym_dpr.envs.DPR_SuperAgent import SuperCircularBot
from gym_dpr.envs.DPR_World import World


class DPRSingleEnv(gym.Env):
    '''Gym Environment for Centralized Particle Robot Control'''
    metadata = {'render.modes': ['human']}

    def __init__(self, numBots, worldClass=World, botClass=CircularBot, superBotClass=SuperCircularBot,
                 discreteActionSpace=False, continuousAction=False,
                 goalFrame=True,
                 rewardFunc="piecewise",
                 randomSeed=0,
                 fixedStart=False, fixedGoal=True,
                 fixedStartCoords=None, fixedGoalCoords=(0, 0),
                 polarStartCoords=True, polarGoalCoords=False,
                 transformRectStart=(0, 0), transformRectGoal=(0, 0),
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
        self.superAgentClass = superBotClass
        self.agent = None

        self.start = None
        self.goal = None

        self.discreteActionSpace = discreteActionSpace
        self.continuousAction = continuousAction
        if continuousAction:
            self.action_space = spaces.Box(low=0, high=math.pi/2, shape=(numBots, ))
        else:
            if discreteActionSpace:
                self.action_space = spaces.Discrete((2 ** self.num) - 1)
            else:
                self.action_space = spaces.MultiBinary(numBots)

        self.goalFrame = goalFrame
        if manipulationTask:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(4 * numBots + 4, ))
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(4 * numBots, ))

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
            if rewardFunc == "Piecewise":
                self.rewardFunc = self.piecewiseVecReward
            elif rewardFunc == "Vec reward":
                self.rewardFunc = self.vecReward
            elif rewardFunc == "Inverse displacement":
                self.rewardFunc = self.positionReward
            else:
                self.rewardFunc = self.piecewiseVecReward

        self.fixedStart = fixedStart and (fixedStartCoords != None)
        self.fixedGoal = fixedGoal and (fixedGoalCoords != None)
        self.polarStartCoords = polarStartCoords
        self.polarGoalCoords = polarGoalCoords
        self.transformRectStart = transformRectStart
        self.transformRectGoal = transformRectGoal

        if self.fixedStart:
            self.startPos = fixedStartCoords
        else:
            self.startPos = None

        if self.fixedGoal:
            self.goalPos = fixedGoalCoords
        else:
            self.goalPos = None

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
        super(DPRSingleEnv, self).__init__()

    def positionReward(self):
        '''
        Basic reward function

        :return: inverse of the distance between the Center of Mass(COM) of the particle robot(DPR) system and the goal
        '''
        return 1. / np.linalg.norm(self.agent.getCOM() - self.goal)

    def vecReward(self):
        '''
        goalVec is a vector representing the ideal trajectory (between the COM and the goal)
        actualVec is a vector representing the trajectory of the COM of the DPR system

        :return: length of the projection of actualVec on goalVec
        '''
        goalVec = pymunk.vec2d.Vec2d(self.goal[0] - self.agent.prevCOM[0], self.goal[1] - self.agent.prevCOM[1])
        actualVec = self.agent.currCOM - self.agent.prevCOM
        return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec))

    def piecewiseVecReward(self):
        '''
        same as vecReward() except when the COM is within a threshold distance,
        then the reward is multiplied by (threshold distance / distance to goal).

        :return: length of the projection of actualVec on goalVec,
        possibly increased by a factor inversely proportional to the distance between the COM and goal
        '''
        goalVec = pymunk.vec2d.Vec2d(self.goal[0] - self.agent.prevCOM[0], self.goal[1] - self.agent.prevCOM[1])
        actualVec = self.agent.currCOM - self.agent.prevCOM
        dist = np.linalg.norm(self.agent.getCOM() - self.goal)
        if dist < self.radiusLower:
            return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec)) * (self.radiusLower / dist)
        else:
            return np.linalg.norm(actualVec) * math.cos(goalVec.get_angle_between(actualVec))

    def updateObjectObs(self):
        '''
        Updates target object coordinates

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

    def actionDecimalToBinary(self, decimalAction):
        '''
        Function to change decimal (discrete) action space into multi-binary action space

        :param decimalAction: base-10 action
        :return: action converted to multi binary
        '''
        return [int(char) for char in bin(decimalAction).replace("0b", "").zfill(self.num)]

    def observations(self):
        '''
        Gets an array of positions and velocities of every particle robot and scales them
        If the environment is set to be the manipulation task, the position and velocities of the object is also recorded

        :return: a (n * 4, ) or (n * 4 + 4, ) shaped array containing the x, y positions and velocities
        of each particle robot (sorted by bot index) and possibly manipulated object.
        '''
        def scale(vl, factor=1000.):
            v = np.array(vl)
            return list(v / factor)

        if self.goalFrame:
            obs = self.agent.observeSelf(self.goal)
        else:
            obs = self.agent.observeSelf((0, 0))
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
        bot = self.superAgentClass(self.num, startPos, continuousAction=self.continuousAction, deadIxs=deadIxs)
        self.world.addSuperAgent(bot)
        self.agent = bot
        self.start = startPos
        self.goal = goalPos

    def polar2rect(self, coords):
        '''
        Converts polar coordinates to cartesian coordinates

        :param coords: Polar coordinates
        :return: Cartesian coordinates
        '''
        x = coords[0] * math.cos(coords[1])
        y = coords[0] * math.sin(coords[1])
        return (x, y)

    def initializeObject(self, start, goal):
        '''
        Function to initialize target object position. Places the object adjacent to the Particle Robot system

        :param start: Particle Robot COM start position
        :param goal: Object goal location
        :return: Object start position
        '''
        BOT_DIAMETER = 2 * DPR_ParticleRobot.BOT_RADIUS + DPR_ParticleRobot.PADDLE_WIDTH + DPR_ParticleRobot.PADDLE_LENGTH
        radius = (self.dim / 2) * BOT_DIAMETER + (self.objectDims[1] / 4)
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        l = max(abs(dx), abs(dy))
        s = min(abs(dx), abs(dy))
        side = radius * s / l
        if dx == 0:
            x = start[0]
            y = start[1] + (radius * (dy / abs(dy)))
        elif dy ==0:
            x = start[0] + (radius * (dx / abs(dx)))
            y = start[1]
        elif abs(dx) > abs(dy):
            x = start[0] + (radius * (dx / abs(dx)))
            y = start[1] + (side * (dy / abs(dy)))
        else:
            x = start[0] + (side * (dx / abs(dx)))
            y = start[1] + (radius * (dy / abs(dy)))
        return x, y

    def reset(self, startPos=None, goalPos=None,
              transformRectStart=(0, 0), transformRectGoal=(0, 0),
              deadIxs=None,
              objectType=None, objectPos=None, objectDims=None,
              gate=False):
        '''
        Resets and initializes world

        :param startPos: start position
        :param goalPos: goal position
        :param transformRectStart: apply transformation on start?
        :param transformRectGoal: apply transformation on goal?
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
                    start = self.polar2rect((r, phi))
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

        start = (start[0] + transformRectStart[0] + self.transformRectStart[0],
                  start[1] + transformRectStart[1] + self.transformRectStart[1])

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

        goal = (goal[0] + transformRectGoal[0] + self.transformRectGoal[0],
                 goal[1] + transformRectGoal[1] + self.transformRectGoal[1])

        self.world.drawPoint(goal)

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
            elif self.objectPos != None:
                objPos = self.objectPos
            else:
                objPos = self.initializeObject(start, goal)

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

        self.COMtrajectory = []
        self.objectTrajectory = []
        self.rewards = []
        if self.recordInfo:
            info = {"COM trajectory": self.COMtrajectory, "Object trajectory": self.objectTrajectory,
                "Reward": self.rewards}
        else:
            info = {}

        return self.observations()

    def render(self, mode='human'):
        '''
        Uses Pygame for visualization

        :param mode: Just human for now
        :return:
        '''
        if self.visualizer != None:
            self.visualizer.viz(timestep=self.steps, world=self.world)

    def wavePolicy(self):
        '''
        Hand-crafted baseline algorithm. Particle robots recieve an index based to distance to the goal
        (closer distance, earlier index), and the index determines the particle robot's location in the
        expansion cycle.

        :return: Array of sequential actions
        '''
        BOT_DIAMETER = 60
        cycleIx = []
        dists = []

        def lineThrough2Points(p, q):
            a = p[1] - q[1]
            b = q[0] - p[0]
            c = p[1] * q[0] - p[0] * q[1]
            return (a, b, -c)

        def distPoint2Line(p, line):
            if line[0] == 0 and line[1] == 0:
                return np.linalg.norm(p)
            return abs((line[0] * p[0] + line[1] * p[1] + line[2])) / (math.sqrt(line[0] * line[0] + line[1] * line[1]))

        cx, cy = self.agent.getCOM()
        gx, gy = self.goal
        px = (((gy - cy)**2)/(gx-cx)) + gx
        py = cy
        p2 = pymunk.vec2d.Vec2d((px, py))
        perpLine = lineThrough2Points(self.goal, p2)
        for bot in self.agent.particles:
            dists.append(distPoint2Line(bot.body.position, perpLine))

        lower = min(dists)
        for dist in dists:
            ix = ((dist - lower) // (BOT_DIAMETER)) + 1
            cycleIx.append(int(ix))
        lastAction = max(cycleIx)
        actionArray = np.zeros((lastAction, self.agent.numBots))
        for botIx, ix in enumerate(cycleIx):
            actionArray[ix - 1][botIx] = 1
        return lastAction, actionArray


    def step(self, action):
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

        :param action: particle robot action
        :return: world observations, scalar reward, environment state, information
        '''
        self.steps += 1

        # 1. observations
        self.agent.updateCOM()
        if self.manipulationTask:
            self.updateObjectObs()
        observations = self.observations()

        # 2. react/create magnets
        self.world.botReact()
        # self.world.addMagnets()

        # 3. take actions
        if self.discreteActionSpace:
            action = self.actionDecimalToBinary(action)
        self.agent.actionAll(action)

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
            self.COMtrajectory.append(self.agent.getCOM())
            if self.manipulationTask:
                self.objectTrajectory.append(self.manipulatedObject.body.position)
            self.rewards.append(rewards)
            info = {"COM trajectory": self.COMtrajectory, "Object trajectory": self.objectTrajectory, "Reward": self.rewards}
        else:
            info = {}

        return observations, rewards, done, info
