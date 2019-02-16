import numpy as np

class Gridworld:
	
	def __init__(self, **kwargs):
		self.wind = kwargs.get("wind", (0, 0, 0, -1, -1, -1, -2, -2, -1, 0))
		self.H = kwargs.get("H", 7)
		self.W = kwargs.get("W", 10)
		self.start_pos = kwargs.get("start_pos", (3,0))
		self.end_pos = kwargs.get("end_pos", (3,7))
		self.stochastic = kwargs.get("stochastic", False)
		self.variance = kwargs.get("variance", 0)
		
	# If you haven't reached the end, reward is always -1		
	def _get_reward(self, S):
		return 0 if (S == self.end_pos) else -1
		
		
class Agent:
	
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	UP_LEFT = 4
	UP_RIGHT = 5
	DOWN_LEFT = 6
	DOWN_RIGHT = 7
	ACTIONS = (UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
	
	def __init__(self, start_pos, H, W, num_A, **kwargs):
		self.eps = kwargs.get("eps", 0.1)
		self.alpha = kwargs.get("alpha", 0.5)
		self.gamma = kwargs.get("eps", 0.1)
		self.Q = np.zeros((H,W,num_A))
		self.best_Q = self.Q
		self.path = []
		self.G_best = []
		self.G_list = []
		self.policy = np.zeros((H,W))
		self.start = start_pos
		
		self.actions = kwargs.get("actions", [0, 1, 2, 3])
		self.num_A = num_A
	
	# eps-greedy policy based on current Q
	def _get_action(self, S, train=True):
		
		if (np.random.rand() < self.eps) and train: 	# occasionally select a random action	
			selection = self.actions
			
		else:	# get a set of actions tied for the max value for this state 
			vals = self.Q[S[0], S[1], :]
			selection = [a for a, v in enumerate(vals) if v == np.max(vals)]
		
		return np.random.choice(selection)
	
	# Update Q based on R, future R and curr Q
	def _get_Q_update(self, S, A, S_next, A_next, R):
		
		this_Q = self.Q[ S[0]	  , S[1]	 , A	  ]
		next_Q = self.Q[ S_next[0], S_next[1], A_next ]
		
		# exponentially decaying average of all Q vals for this state
		return (1-self.alpha) * this_Q + self.alpha * ( R + self.gamma*next_Q )
	
	# ensure you don't fall outside our gridworld
	@staticmethod
	def _clip(val, min, max):
		return np.maximum( min, np.minimum(val, max) )
	
	# Gets the next state based on curr S and selected A, in this world
	def _get_next_state(self, S, A, world):
	
		if world.stochastic:
			mod = np.random.choice([-world.variance, 0, world.variance])
		else:
			mod = 0
	
		if A == Agent.UP:
			v = self._clip( S[0] - 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = S[1]
			
		elif A == Agent.DOWN:
			v = self._clip( S[0] + 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = S[1]
			
		elif A == Agent.LEFT:
			v = self._clip( S[0]     + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] - 1			   				, 0, world.W - 1 )
			
		elif A == Agent.RIGHT:
			v = self._clip( S[0] 	+ world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] + 1			   				, 0, world.W - 1 )
			
		elif A == Agent.UP_LEFT:
			v = self._clip( S[0] - 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] - 1			   				, 0, world.W - 1 )
			
		elif A == Agent.UP_RIGHT:
			v = self._clip( S[0] - 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] + 1			   				, 0, world.W - 1 )
		
		elif A == Agent.DOWN_LEFT:
			v = self._clip( S[0] + 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] - 1	 		   				, 0, world.W - 1 )
			
		elif A == Agent.DOWN_RIGHT:
			v = self._clip( S[0] + 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] + 1			   				, 0, world.W - 1 )
			
		else:
			raise ValueError("A is not valid")

		return (v, h)
	
	def _set_Q_to_default(self):
		
		Q = np.zeros((H, W, NUM_A)) 
	
		Q[:,:,0] = [ 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 	]

		Q[:,:,1] = [ 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] ,
						[0, 0, 0, 0, 0, 0, 0, 1, 0, 1] ,
						[0, 0, 0, 0, 0, 0, 0, 1, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 	]
						
		Q[:,:,2] = [ 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 1, 0] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 1, 1] ,
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 	]
						
		Q[:,:,3] = [ 	[1, 1, 1, 1, 1, 1, 1, 1, 1, 0] ,
						[1, 1, 1, 1, 1, 1, 1, 1, 1, 0] ,
						[1, 1, 1, 1, 1, 1, 1, 1, 1, 0] ,
						[1, 1, 1, 1, 1, 1, 1, 1, 1, 0] ,
						[1, 1, 1, 1, 1, 1, 1, 0, 0, 0] ,
						[1, 1, 1, 1, 1, 1, 1, 0, 0, 0] ,
						[1, 1, 1, 1, 1, 1, 1, 1, 1, 0] 	]
	
		self.Q = Q
	
	# Runs a single episode in the provided world
	def run_episode(self, world, move_timeout=1000, train=True, print_moves=100):
	
		moves = 1		# moves made this episode
		G = 0			# total returns of episode
		path = []		# ordered list of states visited this episode
		
		S = self.start
		A = self._get_action(S, train=train)
		path.append(S)	# Append the starting state		
		
		while S != world.end_pos: # While not at end pos
			
			S_next 	= self._get_next_state(S, A, world) 	# marker for next state			
			R 		= world._get_reward(S_next) 		# Reward of this (S,A) transition
			A_next 	= self._get_action(S_next, train=train) # marker for next action
			
			if moves % print_moves == 0:
				print(f"Move: {moves}, S: {S}, A: {A}, S': {S_next}, A': {A_next}")
				# print()
				# print(np.round(np.max(Q, axis=2), decimals=1))
				# print()
			
			if (train):	# Only update our Q values if training
				self.Q[S[0], S[1], A] = self._get_Q_update(S, A, S_next, A_next, R)
				
			G += R		# Accumulate rewards for episode
			S = S_next	
			A = A_next
			
			path.append(S)	# Append next state (now this state) to episode
			
			if moves > move_timeout:	# If we don't finish for long enough, give up and restart
				print(f"Episode failed...\n")
				break;

			moves += 1
			
		return G, path
		
	# Trains the episode by running specified episodes and occasionally printing to console
	def train_agent(self, world, episodes=100, move_timeout=1000, print_moves=100):
		
		best_G = -move_timeout
		for episode in range(episodes):
			
			G, path = self.run_episode(world, move_timeout=move_timeout, train=True, print_moves=print_moves)
			
			self.G_list.append(G)
			
			if G > best_G:
				print(f"Best run improved to {-G} moves")
				best_G = G
				self.best_Q = self.Q
	
		self.path = path # store most recent path taken
	
	# Maps directions to arrows (strings) for visualization 	
	@staticmethod
	def _get_dir_str(world, num):

		if (world.H,world.W) == world.end_pos:
			return "GG"
		if (world.H,world.W) == world.start_pos:
			return "SS"

		if num == Agent.UP:
			return "^^"
		if num == Agent.DOWN: 
			return "VV"
		if num == Agent.LEFT:
			return "<-"
		if num == Agent.RIGHT:
			return "->"
		if num == Agent.UP_LEFT:
			return "|-"
		if num == Agent.UP_RIGHT:
			return "-|"
		if num == Agent.DOWN_LEFT:
			return "|_"
		if num == Agent.DOWN_RIGHT:
			return "_|"
	
	# Returns a visualization of the policy (chars or nums)
	def get_policy(self, world=None, visual=False):
		
		policy = []

		# Loop through all states
		for h in range(world.H):
			policy.append([]) # create a list inside list (needed because state space is 2D)
			for w in range(world.W):
				# Store the most action with highest Q value for this state
				if visual:
					policy[h].append( self._get_dir_str(world, np.argmax(self.Q[h,w,:]) ) )
				else:
					policy[h].append( np.argmax(self.Q[h,w,:]) )

		return policy
		