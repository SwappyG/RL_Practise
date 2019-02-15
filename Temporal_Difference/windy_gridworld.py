import numpy as np
import matplotlib.pyplot as plt

# Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)

NUM_A = 4
H = 7
W = 10

START_POS = (3, 0)
END_POS = (3, 7)

WIND = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)

GAMMA = 1

PRINT_AFTER_MOVES = 10000
PRINT_AFTER_EPISODES = 10
KEEP_TRAINING_CHECK = 100
MOVE_TIMEOUT = 100000

EPS = 0.1
ALPHA = 0.5

# ensure you don't fall outside our gridworld
def clip(val, min, max):
	# print(f"val {val}  min {min}  max {max}")
	return np.maximum( min, np.minimum(val, max) )

# get next state based on curr action, state, and wind
def get_next_state(S, A):

	if A == UP:
		v = clip( S[0] + 1 + WIND[S[1]], 0, H-1 )
		return (v , S[1])
		
	elif A == DOWN:
		v = clip( S[0] - 1 + WIND[S[1]], 0, H-1 )
		return (v , S[1])
		
	elif A == LEFT:
		v = clip( S[0] + WIND[S[1]], 0, H-1 )
		h = clip( S[1] - 1, 		 0, W-1 )
		return (v, h)
		
	elif A == RIGHT:
		v = clip( S[0] + WIND[S[1]], 0, H-1 )
		h = clip( S[1] + 1, 		 0, W-1 )
		return (v, h)
		
	else:
		raise ValueError("A is not valid")

# eps-greedy policy based on current Q
def get_action(Q, S, train=True):
	
	# occasionally select a random action
	if np.random.rand() < EPS: 
		return np.random.choice(ACTIONS)
		
	else:
		# return np.argmax(Q[S[0], S[1], :])
	
		# get all the values for this state pair
		vals = Q[S[0], S[1], :]
		
		# return the action with max value for this S, breaking ties randomly
		return np.random.choice([a for a, v in enumerate(vals) if v == np.max(vals)])
		
# def get_action(Q, state, train=False):

	# if np.random.binomial(1, EPS) == 1:
		# return np.random.choice(ACTIONS)
	# else:
		# # get all the values for this state pair
		# vals = Q[state[0], state[1], :]
		
		# # return the action with max value for this S, breaking ties randomly
		# return np.random.choice([a for a, v in enumerate(vals) if v == np.max(vals)])
		
# If you haven't reached the end, reward is always -1		
def get_reward(new_state):
	
	if new_state == END_POS:
		return 0
	else:
		return -1

# Update Q based on R, future R and curr Q
def update_Q(Q, S, A, S_next, A_next, R):
	
	this_Q = Q[S[0], S[1], A]
	next_Q = Q[S_next[0], S_next[1], A_next]
	
	return this_Q + ALPHA * ( R + GAMMA * next_Q - this_Q )
	
# Run an episode from some start state and initial action
def run_episode(Q, S, A, train=True):
	
	moves = 1
	total_R = 0
	path = []
	path.append(S)
	
	while S != END_POS: # While not at end pos
		
		S_next = get_next_state(S, A) # marker for next state			
		R = get_reward(S_next) # Reward of this (S,A) transition
		A_next = get_action(Q, S, train=train) # marker for next action
		
		if moves % PRINT_AFTER_MOVES == 0:
			print(f"Move: {moves}, S: {S}, A: {A}, S': {S_next}, A': {A_next}")
			print()
			print(np.round(np.max(Q, axis=2), decimals=1))
			print()
		
		if (train):
			Q[S[0], S[1], A] = update_Q(Q, S, A, S_next, A_next, R)
			
		total_R += R
		S = S_next
		A = A_next
		
		path.append(S)
		
		if moves > MOVE_TIMEOUT:
			print(f"Episode failed...\n")
			break;

		moves += 1
		
	return total_R, path

def keep_training(episode, R):

	print(f"Trained for {episode} episodes, with R_avg: {R}\n")
	return input(f"Keep training? Press any key to continue, or 'n' to stop:\n") == 'n'

def init_R_avg():

	Q = np.zeros((H, W, NUM_A))
	S = start # marker for current state
	A = get_action(Q, S, NUM_A, eps) # marker for current action
	
	R_init, _ = run_episode(Q, S, A, end, eps, alpha, train=False)
	
	return R_init 
	
def main():
	
	Q = np.zeros((H, W, NUM_A)) # Action-value function
	print(Q[:, :, 0])
	
	episode = 1
	
	# Initialize Reward before training for baseline
	R_avg, _ = run_episode(Q, START_POS, get_action(Q, START_POS))
	R_best = R_avg
	R_list = []
	
	Q = np.zeros((H, W, NUM_A)) # reset Q from R baseline calc before starting
	Q_best = Q
	while True:

		if episode % PRINT_AFTER_EPISODES == 0:
			print(f"Running episode {episode}")
		
		# Get the initial state and first action
		S = START_POS 
		A = get_action(Q, S)
		
		# Run an episode from first (S,A) pair
		total_R, _ = run_episode(Q, S, A)
		
		R_list.append(-total_R)
		
		if total_R > R_best:
			R_best = total_R
			Q_best = Q
		
		episode += 1
		
		if episode % KEEP_TRAINING_CHECK == 0:
			# eps = eps/2
			# alpha = alpha/2
			if keep_training(episode, -R_best):
				print(f"Stopping now..")
				break
			
	policy = np.argmax(Q_best, axis=2)
	R, path = run_episode( Q, START_POS, get_action(Q, S), train=False )
	
	print(f"The current policy is:\n\n")
	print(policy)
	print(f"\nMoves to goal: {-R}")
		
	print()
	y,x = zip(*path)
	fig, ax = plt.subplots(2,1)
	ax[0].plot(x,y)
	ax[1].plot(R_list)
	# ax[1].set_yscale('log')
	plt.show()



if __name__=="__main__":
	main()