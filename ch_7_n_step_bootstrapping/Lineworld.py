import numpy as np

class Lineworld:
	
	RIGHT_REWARD = 1
	LEFT_REWARD = 1
	STEP_REWARD = -1
	TIMEOUT_REWARD = -1000
	
	LEFT = 0
	RIGHT = 1
	
	def __init__(self, **kwargs):
		self.L = kwargs.get("L", 19)
		self.start_pos = kwargs.get("start_pos", 2)
		self.end_pos = kwargs.get("end_pos", (0,self.L-1))
		
		
	# If you haven't reached the end, reward is always -1		
	def get_reward(self, S, timeout=False):
	
		if timeout:									return Lineworld.TIMEOUT_REWARD
		elif S == self.end_pos[Lineworld.RIGHT]:	return Lineworld.RIGHT_REWARD
		elif S == self.end_pos[Lineworld.LEFT]:		return Lineworld.LEFT_REWARD	
		else:										return Lineworld.STEP_REWARD
	
	def get_image(self):
		
		world_image = np.zeros((self.H, self.W))
		
		for ii in self.hazard:
			world_image[ii] = 3
		
		for col,val in enumerate(self.wind):
			world_image[:,col] = val
		
		world_image[self.start_pos] = 4
		world_image[self.end_pos] = 5
		
		
		return world_image