import numpy as np
import matplotlib.pyplot as plt

def generate_bandit(k):

	return np.random.normal(loc=0, scale=1, size=(k))

def update_bandit(bandit, mean=0, var=0.01):
	
	return bandit + np.random.normal(loc=mean, scale=var, size=(len(bandit)))

def get_reward(mean, var=1):
	
	return np.random.normal(loc=mean, scale=var)

def get_action(Q, epsilon=0):	
	
	if np.random.rand() < epsilon:
		return np.random.randint(len(Q))
	else:
		return np.argmax(Q)
		
def main():

	NUM_BANDITS = 2000
	NUM_ITERATIONS = 1000
	PRINT_EVERY = 100
	K = 10
	EPSILON = [0, 0.01, 0.1, 0.5]
	
	for eps in EPSILON:
	
		print(f"\nRunning Eps: {eps}")
	
		avg_rewards = np.zeros(NUM_ITERATIONS)
		
		for run in range(NUM_BANDITS):
		
			rewards = np.zeros(NUM_ITERATIONS)

			if run % PRINT_EVERY == 0:
				print(f"Run {run+1} of {NUM_BANDITS}")
				
			bandit = generate_bandit(K)
			Q = np.zeros(K)
			actions_taken = np.zeros_like(Q)
			rewards_gotten = np.zeros_like(Q)
			
			# print(f"Bandit means: {np.around(bandit, 3)}\n")
			
			for iter in range(NUM_ITERATIONS):
				
				this_action = get_action(Q, eps)
				this_reward = get_reward(bandit[this_action])
				
				Q[this_action] += 1/(iter+1) * (this_reward - Q[this_action])
				
				rewards[iter] = this_reward
				
				bandit = update_bandit(bandit, mean=0, var=0.01)

			avg_rewards += 1/(run+1) * (rewards - avg_rewards)

		plt.plot(avg_rewards, label=str(eps))
		
	plt.legend()
	plt.show()
	
	
	















if __name__ == "__main__":

	main()