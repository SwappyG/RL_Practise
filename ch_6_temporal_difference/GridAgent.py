import numpy as np
import matplotlib.pyplot as plt
from Gridworld import Gridworld
		
class GridAgent:
	
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	UP_LEFT = 4
	UP_RIGHT = 5
	DOWN_LEFT = 6
	DOWN_RIGHT = 7
	ACTIONS = (UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
	
	SARSA = 0
	EXPECTED_SARSA = 1
	Q_LEARNING = 2
	METHODS = [SARSA, EXPECTED_SARSA, Q_LEARNING]
	
	def __init__(self, start_pos, H, W, num_A, **kwargs):
		self.eps = kwargs.get("eps", 0.1)
		self.alpha = kwargs.get("alpha", 0.5)
		self.alpha_ramp = kwargs.get("alpha_ramp", 0.99)
		self.gamma = kwargs.get("gamma", 1)
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
	def _get_action(self, S, train=True, all_best=False):
		
		if (np.random.rand() < self.eps) and train: 	# occasionally select a random action	
			selection = self.actions
			
		else:	# get a set of actions tied for the max value for this state 
			vals = self.Q[S[0], S[1], :]
			selection = [a for a, v in enumerate(vals) if v == np.max(vals)]
		
		if all_best: # If function asked to return all best actions
			return selection
		
		# Pick randomly from best actions
		return np.random.choice(selection)
	
	# Finds the expected value of state
	def _get_weighted_Q(self, S):
	
		Q_val = 0
		selection = self._get_action(S, train=False, all_best=True) 	# List of tied best actions for this state
		prob_exploration = self.eps / self.num_A	# Probability of exploring and selecting particular action
		prob_exploitation = (1 - self.eps) / len(selection)	# Probability of exploiting and selecting particular action

		for act in self.actions: # For every possible action
			if act in selection: # If the action is tied for best Q value
				Q_val += (prob_exploitation + prob_exploration) * self.Q[ S[0], S[1], act]
			
			else:
				Q_val +=  prob_exploration * self.Q[ S[0], S[1], act]
			
		return Q_val
			
	# Update Q based on R, future R and curr Q
	def _get_Q_update(self, S, A, S_next, A_next, R, method=None):
		
		if method==None: method=GridAgent.SARSA
		
		this_Q = self.Q[ S[0], S[1], A ]
		
		if method == GridAgent.SARSA:
			next_Q = self.Q[ S_next[0], S_next[1], A_next ]
			
		elif method == GridAgent.Q_LEARNING:
			best_A = self._get_action(S_next, train=False)
			next_Q = self.Q[ S_next[0], S_next[1], best_A ]
			
		elif method == GridAgent.EXPECTED_SARSA:
			next_Q = self._get_weighted_Q(S_next)
				
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
	
		if A == GridAgent.UP:
			v = self._clip( S[0] - 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = S[1]
			
		elif A == GridAgent.DOWN:
			v = self._clip( S[0] + 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = S[1]
			
		elif A == GridAgent.LEFT:
			v = self._clip( S[0]     + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] - 1			   				, 0, world.W - 1 )
			
		elif A == GridAgent.RIGHT:
			v = self._clip( S[0] 	+ world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] + 1			   				, 0, world.W - 1 )
			
		elif A == GridAgent.UP_LEFT:
			v = self._clip( S[0] - 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] - 1			   				, 0, world.W - 1 )
			
		elif A == GridAgent.UP_RIGHT:
			v = self._clip( S[0] - 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] + 1			   				, 0, world.W - 1 )
		
		elif A == GridAgent.DOWN_LEFT:
			v = self._clip( S[0] + 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] - 1	 		   				, 0, world.W - 1 )
			
		elif A == GridAgent.DOWN_RIGHT:
			v = self._clip( S[0] + 1 + world.wind[S[1]] + mod, 0, world.H - 1 )
			h = self._clip( S[1] + 1			   				, 0, world.W - 1 )
			
		else:
			raise ValueError("A is not valid")

		return (v, h)
	
	def set_Q_to_default(self):
		
		Q = self.Q
	
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
	def run_episode(self, world, move_timeout=1000, train=True, print_moves=100, method=None):
	
		if method==None: method = GridAgent.SARSA
	
		moves = 1		# moves made this episode
		G = 0			# total returns of episode
		path = []		# ordered list of states visited this episode
		
		S = self.start
		A = self._get_action(S, train=train)
		path.append(S)	# Append the starting state		
		
		while not (S in world.end_pos): # While not at end pos
						
			S_next 	= self._get_next_state(S, A, world) 	# marker for next state
			R 		= world.get_reward(S_next, timeout=(moves>move_timeout))  # Reward of this (S,A) transition
			
			if world.is_hazard(S_next):		# If agent moves to hazard sq, return to start
				S_next = self.start
			
			A_next 	= self._get_action(S_next, train=train) # marker for next action
			
			if (train):	# Only update our Q values if training
				self.Q[S[0], S[1], A] = self._get_Q_update(S, A, S_next, A_next, R, method=method)
				
			if moves % print_moves == 0:
				print(f"Move: {moves}, S: {S}, A: {A}, S': {S_next}, A': {A_next}")
				
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
	def train_agent(self, world, episodes=100, move_timeout=1000, print_moves=100, ramp_alpha=False, method=None):
		
		if method==None: method = GridAgent.SARSA
			
		best_G = -move_timeout
		for episode in range(episodes):
			
			G, path = self.run_episode(world, move_timeout=move_timeout, train=True, print_moves=print_moves, method=method)
			
			self.G_list.append(G)
			
			if G > best_G:
				print(f"Best run improved to {-G} moves")
				best_G = G
				self.best_Q = self.Q
			
			if ramp_alpha:
				self.alpha *= self.alpha_ramp
	
		self.path = path # store most recent path taken
	
	# Maps directions to arrows (strings) for visualization 	
	@staticmethod
	def _get_dir_str(h,w, start, end, num):

		if (h,w) == end:
			return "GG"
		if (h,w) == start:
			return "SS"

		if num == GridAgent.UP:
			return "^^"
		if num == GridAgent.DOWN: 
			return "VV"
		if num == GridAgent.LEFT:
			return "<-"
		if num == GridAgent.RIGHT:
			return "->"
		if num == GridAgent.UP_LEFT:
			return "|-"
		if num == GridAgent.UP_RIGHT:
			return "-|"
		if num == GridAgent.DOWN_LEFT:
			return "|_"
		if num == GridAgent.DOWN_RIGHT:
			return "_|"
	
	# Returns a visualization of the policy (chars or nums)
	def get_policy(self, world=None, visual=False):
		
		policy = []
		start = world.start_pos
		end = world.end_pos
		
		# Loop through all states
		for h in range(world.H):
			policy.append([]) # create a list inside list (needed because state space is 2D)
			for w in range(world.W):
				# Store the most action with highest Q value for this state
				action = np.argmax(self.Q[h,w,:])
				if visual:
					policy[h].append( self._get_dir_str(h, w, start, end, action) )
				else:
					policy[h].append( action )

		return policy
		