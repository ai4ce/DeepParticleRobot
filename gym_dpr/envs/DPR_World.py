import pymunk
import pymunk.pygame_util 

from gym_dpr.envs.DPR_SuperAgent import SuperCircularBot
import gym_dpr.envs.DPR_ParticleRobot as DPR_ParticleRobot
from gym_dpr.envs.DPR_ParticleRobot import ParticleRobot

import itertools
import math
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point

GROUND_LINEAR_FRICTION = 500
GROUND_ANGULAR_FRICTION = 500
GROUND_VELOCITY_DAMPING = 0.75

GATE_ELASTICITY = 0
GATE_FRICTION = 1

OBJECT_ELASTICITY = 0
OBJECT_FRICTION = 1

JOINT_MAX_FORCE = 1000

class World:
    _dt = 1. / 60
    pymunk_steps_per_frame = 6  # this means action frequency = 1/(_dt*pymunk_steps_per_frame) = 10Hz

    def __init__(self, start=(300, 300), goal=(0, 0), visualizer=None):
        self.start = start
        self.goal = goal

        self.space = pymunk.Space()
        self.superAgents = []
        self.particles = []
        self.magneticForces = []
        self.object = None
        self.balls = []
        self.boxes = []
        self.gates = []
        self.visualizer = visualizer

        self.staticObjects = []
        pymunk.pygame_util.positive_y_is_up = True

        self.inContact = []
        self.edge_index = [[], []]
        self.node_features = []

    def addParticleRobot(self, bot: ParticleRobot):
        '''
        Adds individual particle robot to the world

        :param bot: single particle robot object
        :return:
        '''
        self.space.add(bot.shape, bot.body)
        for paddle in bot.paddles['paddle1']:
            self.space.add(paddle, paddle.body)

        for paddle in bot.paddles['paddle2']:
            self.space.add(paddle, paddle.body)

        self.space.add(*bot.motors['bot1'])
        self.space.add(*bot.motors['12'])
        self.space.add(*bot.joints['bot1'][0], *bot.joints['bot1'][1])
        self.space.add(*bot.joints['12'][0], *bot.joints['12'][1])
        self.particles.append(bot)

        def preSolveFalse(arbiter, space, data):
            return False

        i = bot.botId
        h = self.space.add_collision_handler(i, i)
        h.pre_solve = preSolveFalse

        # simulate ground friction
        pivot = pymunk.PivotJoint(self.space.static_body, bot.body, (0, 0), (0, 0))
        bot.pivot = pivot
        self.space.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = GROUND_LINEAR_FRICTION  # emulate linear friction

        gear = pymunk.GearJoint(self.space.static_body, bot.body, 0.0, 1.0)
        bot.gear = gear
        self.space.add(gear)
        gear.max_bias = 0  # disable joint correction
        gear.max_force = GROUND_ANGULAR_FRICTION

    def addSuperAgent(self, bot: SuperCircularBot):
        '''
        Adds a particle robot superagent to the world

        :param bot: single particle robot superagent object
        :return:
        '''
        self.superAgents.append(bot)
        for bot in bot.particles:
            self.addParticleRobot(bot)

    def generatePairs(self):
        '''
        Finds pairs of particle robots in the world that will exert magnetic attractions to each other
        (distance between must be less than 2 * bot diameter)

        :return: list of pairs
        '''
        pairs = itertools.combinations(self.particles, 2)
        botPairs = [(pair[0], pair[1]) for pair in pairs
                    if np.linalg.norm(pair[0].body.position - pair[1].body.position) < (
                            DPR_ParticleRobot.BOT_RADIUS * 4)]
        return botPairs

    def generateJoints(self, bot1, bot2):
        '''
        Creates a joint to simulate magnetic attraction between two particle robots

        :param bot1: first particle robot
        :param bot2: second particle robot
        :return: "magnetic" joint
        '''
        dists = np.array([np.linalg.norm(paddle.body.position - bot1.shape.body.position)
                          for paddle in bot2.paddles['paddle2']])
        ix = np.argmin(dists)
        bot2paddle = bot2.paddles['paddle2'][ix]

        dists = np.array([np.linalg.norm(paddle.body.position - bot2.shape.body.position)
                          for paddle in bot1.paddles['paddle2']])
        ix = np.argmin(dists)
        bot1paddle = bot1.paddles['paddle2'][ix]

        if len(bot1paddle.body.constraints) == 3 and len(bot2paddle.body.constraints) == 3:
            joint = pymunk.PinJoint(bot1paddle.body, bot2paddle.body)
            joint.max_force = JOINT_MAX_FORCE
            return joint

    def addMagnets(self):
        '''
        Adds joints for every pair of particle robot

        :return:
        '''
        botPairs = self.generatePairs()
        self.inContact = botPairs
        joints = []
        for pair in botPairs:
            mag = self.generateJoints(pair[0], pair[1])
            if mag != None:
                joints.append(mag)
        self.space.add(*joints)
        self.magneticForces = joints

    def removeMagnets(self):
        '''
        Remove all magnets in the world

        :return:
        '''
        self.space.remove(*self.magneticForces)
        self.magneticForces = []

    def botReact(self):
        '''
        Alternate magnet creation. Triggered by individual particle robots

        :return:
        '''
        for bot in self.particles:
            magnets = bot.createAllMagnets(self.particles)
            self.space.add(*magnets)

    def botRemoveMagnets(self):
        '''
        Alternate magnet removal. Triggered by individual particle robots

        :return:
        '''
        for bot in self.particles:
            self.space.remove(*bot.magnets)
            bot.magnets = []

    def addBall(self, pos, mass, radius):
        '''
        Creates a ball that will be the manipulated object by the particle robots

        :param pos: Position of ball
        :param mass: Mass of ball
        :param radius: Radius of ball
        :return: the Pymunk ball object
        '''
        ballBody = pymunk.Body(mass=mass, moment=pymunk.moment_for_circle(mass, 0, radius))
        ballBody.position = pos
        ballShape = pymunk.Circle(ballBody, radius=radius)
        ballShape.elasticity = OBJECT_ELASTICITY
        ballShape.friction = OBJECT_FRICTION
        ballShape.color = (255, 150, 0, 255)
        ballShape.collision_type = 10**10
        self.space.add(ballShape, ballBody)
        self.balls.append(ballShape)

        # simulate ground friction
        pivot = pymunk.PivotJoint(self.space.static_body, ballBody, (0, 0), (0, 0))
        ballShape.pivot = pivot
        self.space.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = GROUND_LINEAR_FRICTION  # emulate linear friction

        gear = pymunk.GearJoint(self.space.static_body, ballBody, 0.0, 1.0)
        ballShape.gear = gear
        self.space.add(gear)
        gear.max_bias = 0  # disable joint correction
        gear.max_force = GROUND_ANGULAR_FRICTION

        return ballShape

    def addBox(self, pos, mass, width, height):
        '''
        Creates a box that will be the manipulated object by the particle robots

        :param pos: Position of box
        :param mass: Mass of box
        :param width: width of ball
        :param height: height of ball
        :return: the Pymunk box object
        '''
        boxBody = pymunk.Body(mass=mass, moment=pymunk.moment_for_box(mass, (width, height)))
        boxBody.position = pos
        boxShape = pymunk.Poly.create_box(boxBody, (width, height))
        boxShape.elasticity = OBJECT_ELASTICITY
        boxShape.friction = OBJECT_FRICTION
        boxShape.color = (255, 150, 0, 255)
        boxShape.collision_type = 10**10
        self.space.add(boxShape, boxBody)
        self.boxes.append(boxShape)

        # simulate ground friction
        pivot = pymunk.PivotJoint(self.space.static_body, boxBody, (0, 0), (0, 0))
        boxShape.pivot = pivot
        self.space.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = GROUND_LINEAR_FRICTION  # emulate linear friction

        gear = pymunk.GearJoint(self.space.static_body, boxBody, 0.0, 1.0)
        boxShape.gear = gear
        self.space.add(gear)
        gear.max_bias = 0  # disable joint correction
        gear.max_force = GROUND_ANGULAR_FRICTION
        return boxShape

    def addObject(self, type, args : list):
        '''
        Adds object for manipulation task

        :param type: "Ball" or "Box"
        :param args: List of object dimensions
        :return: Pymunk shape object
        '''
        if type == 'Ball':
            obj = self.addBall(args[0], args[1], args[2])
        elif type == 'Box':
            obj = self.addBox(args[0], args[1], args[2], args[3])
        else:
            raise ValueError("Value Error no object type: " + type)
        self.object = obj
        return obj


    def getObjectObs(self):
        '''

        :return: gets cartesian coordinates and velocity components of target object
        '''
        x = self.object.body.position[0]
        y = self.object.body.position[1]
        xdot = self.object.body.velocity[0]
        ydot = self.object.body.velocity[1]
        return np.array([x, y, xdot, ydot])

    def removeAll(self):
        '''
        Removes everything from the world

        :return:
        '''
        self.space.remove(*self.space.bodies)
        self.space.remove(*self.space.shapes)
        self.space.remove(*self.space.constraints)

        self.superAgents = []
        self.particles = []
        self.magneticForces = []
        self.balls = []
        self.boxes = []
        self.object = None
        self.gates = []

        self.inContact = []

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

        cx, cy = self.superAgents[0].getCOM()
        gx, gy = self.goal
        px = (((gy - cy) ** 2) / (gx - cx)) + gx
        py = cy
        p2 = pymunk.vec2d.Vec2d(px, py)
        perpLine = lineThrough2Points(self.goal, p2)
        for bot in self.superAgents[0].particles:
            dists.append(distPoint2Line(bot.body.position, perpLine))

        lower = min(dists)
        for dist in dists:
            ix = ((dist - lower) // (BOT_DIAMETER)) + 1
            cycleIx.append(int(ix))
        lastAction = max(cycleIx)
        actionArray = np.zeros((lastAction, self.superAgents[0].numBots))
        for botIx, ix in enumerate(cycleIx):
            actionArray[ix - 1][botIx] = 1
        return lastAction, actionArray

    def step(self, action):
        '''
        1. Gets observations
        2. Simulates magnets
        3. Takes actions
        4. Simulates in pymunk
        5. Resets magnets

        :param timestep: number of timesteps
        :return:
        '''
        # 1. Superagent observation
        observations = self.superAgents[0].observeSelf(self.goal)

        # 2. react (simulate magnetic forces)
        # self.addMagnets()
        self.botReact()

        # 3. take actions
        # action = np.random.randint(2, size=self.superAgents[0].numBots)
        self.superAgents[0].actionAll(action)

        # 4. simulate
        for i in range(self.pymunk_steps_per_frame):
            self.space.step(self._dt)

        # 5. clear magnetic forces
        # self.removeMagnets()
        self.botRemoveMagnets()

        self.edge_index = [[], []]
        for pair in self.generatePairs():
            if np.linalg.norm(pair[0].body.position - self.goal) < np.linalg.norm(pair[1].body.position - self.goal):
                self.edge_index[0].append(pair[0].botId)
                self.edge_index[1].append(pair[1].botId)
            else:
                self.edge_index[0].append(pair[1].botId)
                self.edge_index[1].append(pair[0].botId)

        self.node_features = []
        for particle in self.particles:
            self.node_features.append([particle.targetAngle / (math.pi/2)])

    def run(self, n_steps):
        '''
        Runs the world for n_steps

        :param n_steps: total timesteps
        :return:
        '''
        totalD = 0

        assert(n_steps > 0)
        for i in range(n_steps):
            self.frame_id = i
            if self.visualizer is not None and self.visualizer.viz(i, self) == False:
                break
            self.step(i)

    def drawPoint(self, pos):
        '''
        Creates a static point in the world

        :param pos: position of point
        :return:
        '''
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Poly.create_box(body, (50, 50))
        shape.color = (0, 255, 0, 255)
        self.space.add(shape, body)
        self.staticObjects.append(shape)

    def addGate(self, goal, com, size):
        '''
        Creates a gate in the world. The opening lies in between the start and goal position

        :param goal: goal position
        :param com: particle robot start position
        :param size: width of wall
        :return:
        '''
        mx, my = (goal[0]+ com[0]) / 2, (goal[1] + com[1]) / 2
        def lineThrough2Points(p, q):
            a = p[1] - q[1]
            b = q[0] - p[0]
            c = p[1] * q[0] - p[0] * q[1]
            return (a, b, -c)

        def getPerpLine(p, q):
            a, b, c = lineThrough2Points(p, q)
            mx, my = (p + q[0]) / 2, (p[1] + q[1]) / 2
            d = a * my - b * mx
            return (b, -a, d)

        def gen2points(line):
            a, b, c = line
            if b == 0:
                px = - c / a
                py = -2500

                qx = -c / a
                qy = 2500
                return [(px, py), (qx, qy)]

            px = -2500
            py = (-a * px - c) / b

            qx = 2500
            qy = (-a * qx - c) / b
            return [(px, py), (qx, qy)]

        gate_width = 150
        wall_width = 100
        wall_length = 500

        a, b, c = lineThrough2Points(com, goal)
        if b == 0:
            interiorC1 = c - a * gate_width / 2
            exteriorC1 = c - a * (gate_width / 2 + wall_length)
            interiorC2 = c + a * gate_width / 2
            exteriorC2 = c + a * (gate_width / 2 + wall_length)
        else:
            interiorC1 = c - b * gate_width / 2 * math.sqrt(1 + (a/b) ** 2)
            exteriorC1 = c - b * (gate_width / 2 + wall_length) * math.sqrt(1 + (a / b) ** 2)
            interiorC2 = c + b * gate_width / 2 * math.sqrt(1 + (a/b) ** 2)
            exteriorC2 = c + b * (gate_width / 2 + wall_length) * math.sqrt(1 + (a / b) ** 2)

        r1 = math.sqrt((gate_width / 2)**2 + (wall_width / 2)**2)
        r2 = math.sqrt(((gate_width / 2) + wall_length)**2 + (wall_width / 2)**2)

        p = Point(mx, my)
        p1 = p.buffer(r1).boundary
        p2 = p.buffer(r2).boundary
        l1 = LineString(gen2points((a, b, interiorC1)))
        l2 = LineString(gen2points((a, b, interiorC2)))
        l3 = LineString(gen2points((a, b, exteriorC1)))
        l4 = LineString(gen2points((a, b, exteriorC2)))

        i1 = p1.intersection(l1)
        i2 = p1.intersection(l2)
        i3 = p2.intersection(l3)
        i4 = p2.intersection(l4)

        pair1 = [i1.geoms[0].coords[0], i1.geoms[1].coords[0]]
        pair2 = [i2.geoms[0].coords[0], i2.geoms[1].coords[0]]
        pair3 = [i3.geoms[0].coords[0], i3.geoms[1].coords[0]]
        pair4 = [i4.geoms[0].coords[0], i4.geoms[1].coords[0]]

        b1 = pymunk.Body(body_type=pymunk.Body.STATIC)
        b1.elasticity = GATE_ELASTICITY
        b1.friction = GATE_FRICTION
        s1 = pymunk.Poly(body=b1, vertices=pair1 + pair3)
        self.space.add(s1, b1)
        self.gates.append(s1)

        b2 = pymunk.Body(body_type=pymunk.Body.STATIC)
        b2.elasticity = GATE_ELASTICITY
        b2.friction = GATE_FRICTION
        s2 = pymunk.Poly(body=b2, vertices=pair2 + pair4)
        self.space.add(s2, b2)
        self.gates.append(s2)