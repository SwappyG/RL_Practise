import numpy as np
import matplotlib.pyplot as plt

START_FLAG = 2
END_FLAG = 3
ON_TRACK = 0
OUT_OF_BOUNDS = 1

END_WIDTH_MAX = 6
END_PAD = 7

VIEW_RANGE = 7
MIN_VEL = 1	
MAX_VEL = 5

THRES = 0.7
PRINT_EVERY = 1000
NUM_EPISODES = 10

ACTION_RANGE = 3

GAMMA = 1
EPSILON = 1

X = 0
Y = 1

class Episode:
	def __init__(self):
		self.pos = []
		self.vel = []
		self.action = []
		self.reward = []

def generate_static_track():
	track = [	[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 ],
				[ 2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 ],
				[ 2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1 ],
				[ 2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1 ],
				[ 2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1 ],
				[ 2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1 ],
				[ 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1 ],
				[ 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1 ],
				[ 1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1 ],
				[ 1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,1,1,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,1,1,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,1,1,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,1,1,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,1,1,1,1,1,1,1,1 ],
				[ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,1,1,1,1,1,1,1,1 ]  ]
				
	valid_start_pos = (1, 2, 3, 4, 5)
	
	return np.array(track), valid_start_pos


def generate_track(width, length):
	
	track = np.zeros((width, length), dtype=int)
	
	start_width = np.maximum(width//2, np.random.randint(width))
	track[:start_width,0] = START_FLAG
	track[start_width:,0] = OUT_OF_BOUNDS
	
	end_width = np.minimum(length//2, np.random.randint(END_WIDTH_MAX)+3)
	track[-1, -end_width:] = END_FLAG
	track[-1, :-end_width] = OUT_OF_BOUNDS
	
	out_lim_bottom = start_width
	out_lim_top = 0
	min_thickness_reached = False
	
	for col in range(1, length):
		
		if out_lim_bottom + out_lim_top + 2 > width:
			min_thickness_reached = True
		
		if col < length-end_width:
			
			if np.random.rand() > THRES and not min_thickness_reached:
				out_lim_bottom += 1
		
			track[out_lim_bottom:, col] = OUT_OF_BOUNDS
	
		if col > 2:
			if np.random.rand() > THRES and not min_thickness_reached:
				out_lim_top += 1
				
			track[:out_lim_top, col] = OUT_OF_BOUNDS
	
	# track = np.pad(track, ((END_PAD,END_PAD), (0,END_PAD)), mode="constant", constant_values=1)
	
	return track
		
def random_policy():
	
	return tuple(np.random.randint(low=-1, high=2, size=2))

def update_velocity(vel, action):
	
	# Sum vel and action, clip to MIN_VEL, clip to MAX_VEL, convert to tuple and return
	return tuple( np.minimum(MAX_VEL, np.maximum(MIN_VEL, np.array(vel) + np.array(action))) )

def update_pos(track, pos, vel, valid_start_pos):
	
	new_pos = tuple( np.array(pos) + np.array(vel) )
	reward = -1
	
	try:
		if track[new_pos] == END_FLAG:
			reward = 0
		elif track[new_pos] == OUT_OF_BOUNDS:
			new_pos = (np.random.choice(valid_start_pos) , 0)
			
	except IndexError:
		new_pos = (np.random.choice(valid_start_pos) , 0)
	
	return new_pos, reward
	
def init_episode(valid_start_pos):

	# create an episode object
	episode = Episode()
	
	# initialize the velocity to 0
	episode.vel.append( (0,0) )
	
	# Choice a random start y pos and set x pos to 0
	episode.pos.append( (np.random.choice(valid_start_pos) , 0) )
	
	return episode

def encode_action(a):

	return a[0] + ACTION_RANGE*a[1]

def decode_action(a):
	
	return (a%3, a//3)
	
def run_episode(episode, track, valid_start_pos, Q=None, on_target=False):
	# While the car hasn't found the end flag
	while track[episode.pos[-1]] != END_FLAG:
	
		# print(f"Running step: {len(episode.pos)}")
	
		# Generate deltas for vels using target_policy (learned)
		if on_target:
			x,y = episode.pos[-1]
			vx,vy = episode.vel[-1]
			action = decode_action( target_policy(Q, x, y, vx, vy) )
		
		# Generate deltas for our vels using random policy (behaviour)
		else:
			action = random_policy()
		
		# Update our velocity using the deltas, capping between 0 and 5
		vel = update_velocity(episode.vel[-1], action)
		
		# add the actions and new velocity to the episode
		episode.action.append( encode_action(action) )
		episode.vel.append(vel)
		
		# Get the next position and corresponding reward
		pos, reward = update_pos(track, episode.pos[-1], episode.vel[-1], valid_start_pos)
		
		# Append pos and reward to our episode
		episode.pos.append(pos)
		episode.reward.append(reward)
		
		if len(episode.pos) > 1000:
			break
	
	return episode

def target_policy(Q, x, y, vx, vy):
	
	return np.argmax( Q[ x, y, vx, vy, : ] )
		
	
def main():

	# generate a track
	track, valid_start_pos = generate_static_track()
	
	L, W = track.shape
	Q = np.zeros((L, W, MAX_VEL+1, MAX_VEL+1, ACTION_RANGE*ACTION_RANGE))
	Q_cache = Q
	C = np.zeros_like(Q)
	
	keep_training = True
	episode_count = 0
	
	while keep_training:
	
		if episode_count % (PRINT_EVERY/10) == 0:		
			print(f"Running episode: {episode_count}")
	
		# create an episode instance
		episode = init_episode(valid_start_pos)
		
		# Run an episode and collect (S, A, S', R) transitions
		episode = run_episode(episode, track, valid_start_pos)
		
		G = 0
		W = 1
		for t in reversed(range(len(episode.reward)-1)):
			
			# print(f"time step: {t}")
			
			# Get the reward plus discounted reward to go for this time step
			G = GAMMA * G + episode.reward[t+1]
			
			# Unpack parameters from this transition
			x, y = episode.pos[t]
			vx, vy = episode.vel[t]
			a = episode.action[t]
			
			# Get the current action from our policy
			curr_a = target_policy(Q, x, y, vx, vy)
			
			# Keep our cumulative sum of weights
			C[x,y,vx,vy,a] += W
			
			'''
			# (1-alpha) * Q_curr  +  alpha * G
			# alpha in this case is the importance sampling ratio
			# G is the new guess for expected returns for this state,action pair
			# Q is the existing guess for expected returns for this state,action pair
			# This Equation does a weight average of curr guess of Q and new guess of Q
			# The more times we've seen this state,action pair, the higher C is
			# The higher C is, the less we move from our current guess of Q (alpha is smaller)
			'''
			
			try:
				Q[x,y,vx,vy,a] = Q[x,y,vx,vy,a] + (W/C[x,y,vx,vy,a]) * ( G - Q[x,y,vx,vy,a] )
			except RuntimeWarning:
				print(Q[x,y,vx,vy,a], W, C[x,y,vx,vy,a], G )
				
			new_a = target_policy(Q, x, y, vx, vy)
			
			if curr_a != new_a:
				break
				
			W = W / (ACTION_RANGE*ACTION_RANGE)
		
		episode_count += 1
		
		if episode_count % PRINT_EVERY == 0:
			
			
			print(f"States filled: {np.count_nonzero(Q) / Q.size}")
			
			
		if episode_count % (PRINT_EVERY*10) == 0:
		
			print(f"Running validation check")
			
			# Run an episode on our target_policy
			episode = init_episode(valid_start_pos)
			episode = run_episode(episode, track, valid_start_pos, Q, on_target=True)
			
			# Plot the result of our target_policy run
			fig, ax = plt.subplots()
			ax.imshow(track)
			
			x, y = zip(*episode.pos)
			ax.plot(y, x)
			ax.scatter(y, x)
			
			plt.show()
			
			# Ask the user if they want to keep training or stop
			user_input = input("Press any key to keep training, or 'n' to stop:\n")
			keep_training = ( user_input != "n" )
			
			# Q_cache = Q
			
			print("\n")
			
if __name__=="__main__":
	main()
	
	
# def get_observation(track, pos):

	# min_w = pos[0] - VIEW_RANGE//2
	# max_w = pos[0] + VIEW_RANGE//2 + 1
	
	# min_h = pos[1] + 1
	# max_h = pos[1] + VIEW_RANGE + 1
	
	# observation = track[ min_w : max_w , min_h : max_h ]
	
	# return observation