import numpy as np
import matplotlib.pyplot as plt

# Creates an array of means of size k
def generate_bandit(k, mean=0):
	return np.random.normal(loc=mean, scale=1, size=(k))

# Performs a random walk on the bandits with mean 0 and var provided
def update_bandit(bandit, mean=0, var=0.01):
	return bandit + np.random.normal(loc=mean, scale=var, size=(len(bandit)))

# Samples from a gaussian around the provided mean and var to generate reward
def get_reward(mean, var=1):
	return np.random.normal(loc=mean, scale=var)

# Gradient based approach for updating Q
def update_Q_gradient_ascent(iter, reward, avg_reward, Q, action, alpha=0):

	# Either uses a weight decaying avg or true avg based on ALPHA
	if alpha == 0:
		step = 1/(iter+1)
	else:
		step = alpha
	
	# calculated the softmax distribution - P(a_i) = e^a_i / sum(e^A)
	pi_A = np.exp(Q[action]) / np.sum(np.exp(Q))
	
	# Gradient update based delta reward and current probability of not choosing it
	# If the delta is positive, and the chance of choosing it is low, Q_a goes up
	# If delta is negative and chance of choosing it is high, Q_a goes down
	# 1 - P(A = a_i) amplifies the delta, and is always > 0 by def'n
	Q[action] = Q[action] + step * (reward - avg_reward) * (1 - pi_A)
	
	# Perform gradient descent for the other elements of Q using similar rationale
	mask = np.ones_like(Q, dtype=int)
	mask[action] = 0
	
	Q[mask] = Q[mask] - step * (reward - avg_reward) * pi_A
	
	return Q
	
# Randomly samples from the softmax distribution and returns an action
def get_action(Q):	
	probs = np.exp(Q) / np.sum(np.exp(Q))
	return np.random.choice(len(Q), p=probs)
	
def main():

	# Constants (Not all used)
	NUM_BANDITS = 1000
	NUM_ITERATIONS = 2000
	PRINT_EVERY = 25
	K = 10
	ALPHA = 0.1
	DO_RANDOM_WALK = True
	WALK_VAR = 0.1
	
	# Keeps track of how we're performing in each run, for plotting purposes
	avg_rewards = np.zeros(NUM_ITERATIONS)
	
	# Repeat training multiple times
	for run in range(NUM_BANDITS):

		# Reset our rewards
		rewards = np.zeros(NUM_ITERATIONS)
		avg_reward = 0
	
		if run % PRINT_EVERY == 0:
			print(f"Run {run+1} of {NUM_BANDITS}")
			
		# Initialize our bandits (q_star) and Q (our guess for q_star)
		bandit = generate_bandit(K, mean=4)
		Q = np.ones(K) / K
		
		# Go through multiple iterations, trying to max reward at each instance
		for iter in range(NUM_ITERATIONS):
			
			# Get the action based on our Q and the corresponding reward (sampled from bandits)
			this_action = get_action(Q)
			this_reward = get_reward(bandit[this_action])
			
			# incorporate this into our average rewards
			avg_reward += 1/(iter+1) * (this_reward - avg_reward)
			
			# Perform gradient ascent / descent on Q
			Q = update_Q_gradient_ascent(iter, this_reward, avg_reward, Q, this_action, alpha=ALPHA)
			
			# Save this value for plotting
			rewards[iter] = this_reward
			
			# Perform a random walk on our bandits if flag is set
			if DO_RANDOM_WALK:				
				bandit = update_bandit(bandit, mean=0, var=WALK_VAR)

		avg_rewards += 1/(run+1) * (rewards - avg_rewards)

	name = f"Alpha: {ALPHA}"
	plt.plot(avg_rewards, label=name)
		
	plt.legend()
	plt.show()
	
	
	















if __name__ == "__main__":

	main()