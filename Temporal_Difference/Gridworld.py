import numpy as np

class Gridworld:
	
	STEP_REWARD = -1
	END_REWARD = 0
	HAZARD_REWARD = -100
	TIMEOUT_REWARD = -5000
	
	def __init__(self, **kwargs):
		self.wind = kwargs.get("wind", (0, 0, 0, -1, -1, -1, -2, -2, -1, 0))
		self.H = kwargs.get("H", 7)
		self.W = kwargs.get("W", 10)
		self.start_pos = kwargs.get("start_pos", (3,0))
		self.end_pos = kwargs.get("end_pos", (3,7))
		self.stochastic = kwargs.get("stochastic", False)
		self.variance = kwargs.get("variance", 0)
		self.hazard = kwargs.get("hazard", [])
		
	# If you haven't reached the end, reward is always -1		
	def get_reward(self, S, timeout=False):
	
		if timeout:
			return self.TIMEOUT_REWARD
	
		if S == self.end_pos:
			return Gridworld.END_REWARD
			
		elif self.is_hazard(S):
			return Gridworld.HAZARD_REWARD
			
		else:
			return Gridworld.STEP_REWARD
		
	def is_hazard(self, S):
		if S in self.hazard:
			return True
		else: 
			return False
	
	def get_image(self):
		
		world_image = np.zeros((self.H, self.W))
		
		for ii in self.hazard:
			world_image[ii] = 3
		
		for col,val in enumerate(self.wind):
			world_image[:,col] = val
		
		world_image[self.start_pos] = 4
		world_image[self.end_pos] = 5
		
		
		return world_image