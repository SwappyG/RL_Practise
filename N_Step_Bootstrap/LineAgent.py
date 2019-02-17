import numpy as np

class LineAgent:
	
	LEFT = 0
	RIGHT = 1
	ACTIONS = (LEFT, RIGHT)
	
	SARSA = 0
	Q_LEARNING = 1
	EXPECTED_SARSA = 2
	
	
	def __init__(self, start_pos, L, **kwargs):
		
		self.eps = kwargs.get("eps", 0.1)
		self.alpha = kwargs.get("alpha", 0.5)
		self.alpha_ramp = kwargs.get("alpha_ramp", 0.99)
		self.gamma = kwargs.get("gamma", 1)
		
		self.start = start_pos
		
		
		self.actions = kwargs.get("actions", LineAgent.ACTIONS)
		self.num_A = len(self.actions)
		
		self.Q = np.zeros((L, self.num_A))
		self.V = np.zeros(L)
		self.best_Q = self.Q
		self.best_V = self.V
		
		self.policy = np.zeros(L)
		self.path = []
		
		self.G_best = []
		self.G_list = []
		
	def _get_action(self, S, train=True, all_best=False):
		
		if (np.random.rand() < self.eps) and train: 	# occasionally select a random action	
			selection = self.actions
			
		else:	# get a set of actions tied for the max value for this state 
			vals = self.Q[S, :]
			selection = [a for a, v in enumerate(vals) if v == np.max(vals)]
		
		if all_best: # If function asked to return all best actions
			return selection
		
		# Pick randomly from best actions
		return np.random.choice(selection)
	
	def _get_next_state(self, S, A, world):
		
		if A == LineAgent.LEFT:
			return 0 if (S-1 < 0) else S-1
		elif A == LineAgent.RIGHT:
			return S+1

	# Finds the expected value of state
	def _get_weighted_Q(self, S):
	
		Q_val = 0
		selection = self._get_action(S, train=False, all_best=True) 	# List of tied best actions for this state
		prob_exploration = self.eps / self.num_A	# Probability of exploring and selecting particular action
		prob_exploitation = (1 - self.eps) / len(selection)	# Probability of exploiting and selecting particular action

		for act in self.actions: # For every possible action
			if act in selection: # If the action is tied for best Q value
				Q_val += (prob_exploitation + prob_exploration) * self.Q[S, act]
			
			else:
				Q_val +=  prob_exploration * self.Q[S, act]
			
		return Q_val
			
	# Update Q based on R, future R and curr Q
	def _get_Q_update(self, S, A, S_next, A_next, R, method=None):
		
		if method==None: method=GridAgent.SARSA
		
		this_Q = self.Q[S, A]
		
		if method == LineAgent.SARSA:
			next_Q = self.Q[S_next, A_next]
			
		elif method == GridAgent.Q_LEARNING:
			best_A = self._get_action(S_next, train=False)
			next_Q = self.Q[S_next, best_A ]
			
		elif method == GridAgent.EXPECTED_SARSA:
			next_Q = self._get_weighted_Q(S_next)
				
		# exponentially decaying average of all Q vals for this state
		return (1-self.alpha) * this_Q + self.alpha * ( R + self.gamma*next_Q )
		
		
	# Runs a single episode in the provided world
	def run_episode(self, world, move_timeout=1000, train=True, print_moves=100, method=None, n_step=1):
	
		if method==None: method = LineAgent.SARSA
	
		moves = 1		# moves made this episode
		G = 0			# total returns of episode
		path = []		# ordered list of states visited this episode
		
		S = self.start
		A = self._get_action(S, train=train)
		path.append(S)	# Append the starting state		
		
		while not (S in world.end_pos): # While not at end pos
			# print(world.end_pos)
			S_next 	= self._get_next_state(S, A, world) 	# marker for next state
			R 		= world.get_reward(S_next, timeout=(moves>move_timeout))  # Reward of this (S,A) transition
			
			A_next 	= self._get_action(S_next, train=train) # marker for next action
			
			if (train):	# Only update our Q values if training
				self.Q[S, A] = self._get_Q_update(S, A, S_next, A_next, R, method=method)
				
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
	def train_agent(self, world, episodes=100, move_timeout=1000, print_moves=100, ramp_alpha=False, method=None, n_step=1):
		
		if method==None: method = LineAgent.SARSA
			
		best_G = -move_timeout
		for episode in range(episodes):
			
			G, path = self.run_episode(world, move_timeout=move_timeout, train=True, print_moves=print_moves, method=method, n_step=n_step)
			
			self.G_list.append(G)
			
			if G > best_G:
				print(f"Best run improved to {-G} moves")
				best_G = G
				self.best_Q = self.Q
			
			if ramp_alpha:
				self.alpha *= self.alpha_ramp
	
		self.path = path # store most recent path taken	
		