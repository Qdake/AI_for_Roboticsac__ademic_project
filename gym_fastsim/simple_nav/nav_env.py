import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import math
import random
import logging

import pyfastsim as fs

logger = logging.getLogger(__name__)

default_env = "assets/LS_maze_hard.xml"
#default_env = "assets/example.xml"

sticky_walls = False # default is True in libfastsim... but false in the fastsim sferes2 module -> stick (haha) with that


def sqdist(x,y):
#|x-y|**2 euclidienne  2D

	return (x[0]-y[0])**2+(x[1]-y[1])**2

def dist(x,y):
#  |x-y|  euclidienne
	return math.sqrt(sqdist(x,y))

class SimpleNavEnv(gym.Env):
	def __init__(self,xml_env=None):
		# Load XML
		if not xml_env:
			path = os.path.join(os.path.dirname(__file__), default_env)
		else:
			path = xml_env
		
		# XML files typically have relative names wrt their own path. Make that work
		oldcwd = os.getcwd()
		os.chdir(os.path.dirname(path))
		settings = fs.Settings(path)
		os.chdir(oldcwd)
		
		self.display = None
		self.map = settings.map()
		self.robot = settings.robot()

		
		self.maxVel = 4 # Same as in the C++ sferes2 experiment
		
		lasers = self.robot.get_lasers()
		n_lasers = len(lasers)
		self.maxSensorRange = lasers[0].get_range() # Assume at least 1 laser ranger
		
		self.initPos = self.get_robot_pos()

		
		self.goal = self.map.get_goals()[0] # Assume 1 goal
		self.goalPos=[self.goal.get_x(),self.goal.get_y()]
		self.goalRadius = self.goal.get_diam()/2.
		
		self.observation_space = spaces.Box(low=np.array([0.]*n_lasers + [0.]*2), high=np.array([self.maxSensorRange]*n_lasers + [1.]*2), dtype=np.float32)
		self.action_space = spaces.Box(low=-self.maxVel, high=self.maxVel, shape=(2,), dtype=np.float32)

	def enable_display(self):
		if not self.display:
			self.display = fs.Display(self.map, self.robot)
			self.display.update()

	def disable_display(self):
		if self.display:
			del self.display
			self.display = None

	def get_robot_pos(self):
		pos = self.robot.get_pos()
		return [pos.x(), pos.y(), pos.theta()]

	def get_laserranges(self):
		out = list()
		for l in self.robot.get_lasers():
			r = l.get_dist()
			if r < 0:
				out.append(self.maxSensorRange)
			else:
				out.append(np.clip(r,0.,self.maxSensorRange))
		return out

	def get_bumpers(self):
		return [float(self.robot.get_left_bumper()), float(self.robot.get_right_bumper())]


	def get_all_sensors(self):
	#[d1,d2,...]  list des donnees des sensor
		return self.get_laserranges() + self.get_bumpers()

	def step(self, action):
		# Action is: [leftWheelVel, rightWheelVel]
		[v1, v2] = action
		
		self.robot.move(np.clip(v1,-self.maxVel,self.maxVel),np.clip(v2,-self.maxVel,self.maxVel), self.map, sticky_walls)

		sensors = self.get_all_sensors()
		reward = self._get_reward()

		p = self.get_robot_pos()
		
		#if(sqdist(p,self.roldpos)<0.001**2):
		#	self.still=self.still+1
		#else:
		#	self.still=0
		#self.roldpos=p
		#episode_over = self.still>=self.still_limit
		episode_over = False

		dist_obj = dist(p, self.goalPos)

		return sensors, reward, episode_over, {"dist_obj":dist_obj, "robot_pos":p}


	def _get_reward(self):
		""" Reward is given when close enough to the goal. """
		p = self.get_robot_pos()
		if (dist(p,self.goalPos)<=self.goalRadius):
			return 1.
		else:
			return 0.
		
	def reset(self):
		p = fs.Posture(*self.initPos)
		self.robot.set_pos(p)
		return self.get_all_sensors()

	def render(self, mode='human', close=False):
		if self.display:
			self.display.update()
		pass

	def close(self):
		self.disable_display()
		del self.robot
		del self.map
		pass
