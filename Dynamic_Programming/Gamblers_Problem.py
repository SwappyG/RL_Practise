import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

THRES = 1e-15
P_WIN = 0.4
P_LOSE = 1-P_WIN
DISCOUNT = 1

WIN_AMOUNT = 100

# Returns True if heads, False otherwise, with probability P
def flip_coin(P):
	return np.random.rand() < P
	
def get_action(V):
	return np.argmax(V)

	
def main():

	# State = amount of money the gambler has
	V = 0.001*np.random.rand(WIN_AMOUNT+1)
	V[WIN_AMOUNT] = 1
	V[0] = 0
	
	policy = np.ones(WIN_AMOUNT, dtype=int)
	
	delta = THRES + 1
	iteration = 0
	
	while delta > THRES:
	
		print(f"Value function not yet stable, running iteration {iteration}, delta: {delta}")
	
		iteration += 1
		delta = 0 # Reset our marker for how much V[s] is changing
		
		# For every possible state 
		for s, v_s in enumerate(V):
			
			if s == WIN_AMOUNT:
				continue
			
			if s == 0:
				V[s] = 0
				policy[s] = 0
				continue
			
			v_max = 0	# keeps track of value with best action
			a_max = 0	# keeps track of best action
			
			# For every possible action from this state
			for a in range(1,s+1):
				
				v_next = P_WIN if s+a > WIN_AMOUNT else P_WIN * (DISCOUNT * V[s+a])	# Discounted value of s' if win
				
				v_next += P_LOSE * (DISCOUNT * V[s-a])		# Discounted value of s' if lose
				
				# If this action produced a better value, store it and the action
				if v_next > v_max:
					v_max = v_next
					a_max = a
				
			# Replace value of current state with value calculated from best action for this state
			V[s] = v_max
			
			# Store best action in our policy for this state
			policy[s] = a_max
			
			# Keep track of our max deviation from previous V[s]
			delta = np.maximum(delta, np.abs(v_s - v_max))
			
	
	fig, ax = plt.subplots(1,2)
	
	ax[0].plot(policy)
	ax[1].plot(V)
	
	plt.show()
	
if __name__=="__main__":
	main()