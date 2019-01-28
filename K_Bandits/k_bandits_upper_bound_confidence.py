import numpy as np
import matplotlib.pyplot as plt

def generate_bandit(k):

	return np.random.normal(loc=0, scale=1, size=(k))

def update_bandit(bandit, mean=0, var=0.01):
	
	return bandit + np.random.normal(loc=mean, scale=var, size=(len(bandit)))

def get_reward(mean, var=1):
	
	return np.random.normal(loc=mean, scale=var)

def update_Q(iter, reward, q_i, alpha=0):

	if alpha == 0:
		return q_i + 1/(iter+1) * (reward - q_i)
	else:
		return q_i + alpha * (reward - q_i)
	
def get_action(Q, c, N, iter):	
	
	return np.argmax(Q + c * np.sqrt( np.log(iter)/N ))
		
def main():

	NUM_BANDITS = 200
	NUM_ITERATIONS = 10000
	PRINT_EVERY = 50
	K = 10
	EPSILON = 0.1
	ALPHA = 0
	C = [1, 1.5, 2, 5]
	
	avg_rewards = np.zeros(NUM_ITERATIONS)
	
	for c in C:
	
		for run in range(NUM_BANDITS):

			rewards = np.zeros(NUM_ITERATIONS)
		
			if run % PRINT_EVERY == 0:
				print(f"Run {run+1} of {NUM_BANDITS}")
				
			bandit = generate_bandit(K)
			Q = np.zeros(K)
			N = np.ones(K)
			actions_taken = np.zeros_like(Q)
			rewards_gotten = np.zeros_like(Q)
			
			# print(f"Bandit means: {np.around(bandit, 3)}\n")
		
			for iter in range(NUM_ITERATIONS):
				
				this_action = get_action(Q, c, N, iter)
				this_reward = get_reward(bandit[this_action])
				
				Q[this_action] = update_Q(iter, this_reward, Q[this_action], alpha=ALPHA)
				N[this_action] += 1
				
				rewards[iter] = this_reward
				
				bandit = update_bandit(bandit, mean=0, var=0.01)

			avg_rewards += 1/(run+1) * (rewards - avg_rewards)

		name = f"C: {c}"
		plt.plot(avg_rewards, label=name)
		
	plt.legend()
	plt.show()
	
	
	















if __name__ == "__main__":

	main()