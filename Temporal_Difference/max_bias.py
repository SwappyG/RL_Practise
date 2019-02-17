import numpy as np
import matplotlib.pyplot as plt

LEFT = 0
RIGHT = 1
START_POS = 2
RIGHT_END = 3
LEFT_END = 0

END_REWARD = 0
STEP_REWARD = 0

Q_LEARNING = 0
DOUBLE_Q_LEARNING = 1

ALPHA = 0.5
ALPHA_RAMP = 1
GAMMA = 1
EPS = 0.1
MEAN = -0.1
VARIANCE = 1

EPISODES = 20000

ACTIONS = [(), range(10), (LEFT,RIGHT), ()]

# eps-greedy policy based on current Q
def get_action(Q, S, eps, train=True, all_best=False):
	
	if (np.random.rand() < eps) and train: 	# occasionally select a random action	
		selection = ACTIONS[S]
		
	else:	# get a set of actions tied for the max value for this state 
		vals = Q[S, ACTIONS[S]]
		selection = [a for a, v in enumerate(vals) if v == np.max(vals)]
	
	if all_best: # If function asked to return all best actions
		return selection
	
	# Pick randomly from best actions
	return np.random.choice(selection)

	
def get_next_Q(Q, S, Q2=None, method=Q_LEARNING):

	if method == Q_LEARNING:
		if S in (RIGHT_END, LEFT_END):
			return 0
		
		else:
			return np.max( Q[S, :] )
			
	if method == DOUBLE_Q_LEARNING:
		if S in (RIGHT_END, LEFT_END):
			return 0
			
		else:
			arg = np.argmax( Q[S, :] )
			return Q2[S, arg]
		

def update_Q(this_Q, next_Q, alpha, gamma, R):
	return  (1-alpha) * this_Q + alpha * ( R + gamma*next_Q )
		
# Runs a single episode in the provided world
def run_episode(Q, Q2=None, train=True, method=Q_LEARNING):

	G = 0			# total returns of episode

	S = START_POS
	A = get_action(Q, S, EPS, train=train)
	first_action = A
	flip = False
	
	if A == RIGHT:
		S_next = RIGHT_END
		R = END_REWARD  # Reward of this (S,A) transition
	
		if (train):	# Only update our Q values if training	
			if method==DOUBLE_Q_LEARNING:
				flip = np.random.choice([True, False])
			
			if flip:
				next_Q = get_next_Q( Q2, S_next, Q2=Q, method=method) #flip pos of Q1 and Q2
				Q2[S,A] = update_Q( Q2[S,A], next_Q, ALPHA, GAMMA, R )
			else:
				next_Q = get_next_Q( Q, S_next, Q2=Q2, method=method)				
				Q[S,A] = update_Q( Q[S,A], next_Q, ALPHA, GAMMA, R )
			
		G += R
		
	elif A == LEFT:
	
		# Middle Right State, no reward
		S_next = S-1
		R = STEP_REWARD
		A_next = get_action(Q, S_next, EPS, train=train)
		
		if train:
			if method==DOUBLE_Q_LEARNING:
				flip = np.random.choice([True, False])
			
			if flip:
				next_Q = get_next_Q( Q2, S_next, Q2=Q, method=method) #flip pos of Q1 and Q2
				Q2[S,A] = update_Q( Q2[S,A], next_Q, ALPHA, GAMMA, R )
			else:
				next_Q = get_next_Q( Q, S_next, Q2=Q2, method=method)				
				Q[S,A] = update_Q( Q[S,A], next_Q, ALPHA, GAMMA, R )
		
		G += R
		S = S_next
		A = A_next
		
		# End state, Reward is normal with some mean and var``
		R = np.random.normal(loc=MEAN, scale=VARIANCE)
		S_next = LEFT_END
		
		if train:
			if method==DOUBLE_Q_LEARNING:
				flip = np.random.choice([True, False])
			
			if flip:
				next_Q = get_next_Q( Q2, S_next, Q2=Q, method=method) #flip pos of Q1 and Q2
				Q2[S,A] = update_Q( Q2[S,A], next_Q, ALPHA, GAMMA, R )
			else:
				next_Q = get_next_Q( Q, S_next, Q2=Q2, method=method)				
				Q[S,A] = update_Q( Q[S,A], next_Q, ALPHA, GAMMA, R )
			
		G += R
		
	return G, first_action


# Trains the episode by running specified episodes and occasionally printing to console
def main():
			
	G_list = []
	A_list = []
	Q = np.zeros( (4, np.max( [len(A) for A in ACTIONS] ) ) )
	Q2 = np.zeros( (4, np.max( [len(A) for A in ACTIONS] ) ) )
	
	for episode in range(EPISODES):
		
		G, A = run_episode(Q, Q2=Q2, method=DOUBLE_Q_LEARNING)
		
		N = len(G_list)+1
		
		if G_list == []:
			G_list.append(G)
			A_list.append(A)
		else:
			G_list.append(  G_list[-1] + (G - G_list[-1]) / N  )
			A_list.append(	A_list[-1] + (A - A_list[-1]) / N	) 
		
	
	# Plot the final path that agent took, and its rewards over course of training
	fig, ax = plt.subplots()
	ax.plot(G_list)
	ax.plot(A_list)
	
	print(f"Freq of choosing right: {np.sum(A_list)/len(A_list)}")
	
	Q = np.zeros( (4, np.max( [len(A) for A in ACTIONS] ) ) )
	G_list = []
	A_list = []
	for episode in range(EPISODES):
		
		G, A = run_episode(Q, method=Q_LEARNING)
		
		N = len(G_list)+1
		
		if G_list == []:
			G_list.append(G)
			A_list.append(A)
		else:
			G_list.append(  G_list[-1] + (G - G_list[-1]) / N  )
			A_list.append(	A_list[-1] + (A - A_list[-1]) / N	) 
			
	
	ax.plot(G_list)
	ax.plot(A_list)
	
	plt.show()
	
if __name__=="__main__":
	main()