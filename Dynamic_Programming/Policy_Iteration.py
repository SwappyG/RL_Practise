""" 
Problem Description

Rental Company has 2 locations
Rental requests follow Poisson distribution with Lamda = [3, 4] for each location
Returns also follow Poission, with Lamda = [3, 2]

Up to 5 Cars can be moved from one location to the other over night, cost = $2/car moved
Company makes $10 per rental request, given there are enough cars at location
$0 are made for any additional requests greater than the number of available cars

Determine an optimal policy which maps number of cars to move based on cars at each location to maximize profit

"""


import numpy as np
from scipy.stats import poisson
import math
import matplotlib.pyplot as plt

LAMDA_REQ = [3, 4]
LAMDA_RET = [3, 2]

DISCOUNT = 0.9
MOVE_COST = 2

MAX_CARS = 7 + 1
THRES = MAX_CARS*MAX_CARS*0.5

MAX_MOVE = 4

RENTAL_PRICE = 10

def get_probs(lamda, max):
	v1 = poisson.pmf(range(max), lamda[0])
	v1[-1] = 1 - poisson.cdf( max-1, lamda[0] )

	v2 = poisson.pmf(range(max), lamda[1])
	v1[-1] = 1 - poisson.cdf( max-1, lamda[1] )
	
	M = v1[:, np.newaxis] * v2
	
	return M

PROBS_REQS = get_probs(LAMDA_REQ, MAX_CARS)
PROBS_RETS = get_probs(LAMDA_RET, MAX_CARS)
	
def get_cars(lamda):
	return np.minimum(np.random.poisson(lamda), MAX_CARS)

def get_money(cars, requests):
	return np.sum(np.minimum(request,cars)*10)

def get_state_value(row, col, action, value):
	
	V = 0 # Initialize V
	
	# For every combination of requests that could arrive
	for req1 in range(MAX_CARS):
		for req2 in range(MAX_CARS):
			
			# Determine the reward
			R = -MOVE_COST*np.abs(action) # Cost of moving cars	
			R += np.minimum(row-action, req1) * RENTAL_PRICE # Reward from location 1
			R += np.minimum(col+action, req2) * RENTAL_PRICE # Reward from location 2
			
			# For every combination of returns that could happen
			for ret1 in range(MAX_CARS):
				for ret2 in range(MAX_CARS):
					
					# Get the next states, given this combo of requests and returns
					s1 = np.minimum(MAX_CARS-1, np.maximum(0, row-action-req1) + ret1)
					s2 = np.minimum(MAX_CARS-1, np.maximum(0, col+action-req2) + ret2)
	
					# add weighted ( Reward + discounted (Reward-to-go) ) to value of this state
					V += PROBS_REQS[req1, req2] * PROBS_RETS[ret1, ret2] * (R + DISCOUNT * value[s1, s2])
					
	return V
	

def get_action_bounds(row, col):

	min_action = np.maximum(-MAX_MOVE, -col) # Can't move more than cars avail in Loc 2
	min_action = np.maximum(min_action, row-(MAX_CARS-1)) # Can't have more than 20 in Loc 1
	max_action = np.minimum(MAX_MOVE, row) # Can't move more than cars avail in Loc 1
	max_action = np.minimum(max_action, MAX_CARS-1-col) # Can't have more than 20 in Loc 2

	return min_action, max_action
	
def main():	
		
	print("Initializing policy and value functions to 0s")

	# Initialize the policy and state-value functions
	policy = np.zeros((MAX_CARS, MAX_CARS), dtype=int)
	value = np.zeros((MAX_CARS, MAX_CARS))

	not_stable = True
	unstable_count = 0
	policy_change_count = 0
	
	# Continue to refine while our Value function is still updating
	while not_stable:

		print(f"\nPolicy still unstable, {policy_change_count} changes made, Running update number {unstable_count}\n")

		sweep_count = 0
		delta = THRES+1
		
		# Keep improving our estimate of V while its still changing
		while delta > THRES:
		
			print(f"Value Func not yet converged, delta = {delta}, state sweep number: {sweep_count}")
			print(f"{np.round(value)}")
		
			# Set our change to 0
			delta = 0
			
			# For every state
			for row in range(MAX_CARS):
				for col in range(MAX_CARS):
					
					# print(f"Running state ({row, col}), action: {policy[row,col]}")
					
					# Get the action from our policy and the negative reward for moving cars
					action = policy[row,col]
					
					V = get_state_value(row, col, action, value)
					
					# Find the delta between our calculated V and existing V, keep track of max delta so far
					delta = np.maximum(delta, np.abs(value[row,col] - V))
					
					# Update our existing value function with the new V
					value[row,col] = V
		
			sweep_count += 1

		print(f"Value function has converged, updating policy")

		not_stable = False
		policy_change_count = 0

		# Iterate through all possible states 
		for row in range(MAX_CARS):
			for col in range(MAX_CARS):
			
				print(f"Updating state {row}, {col}")
			
				curr_action = policy[row, col]
				
				best_action = 0
				best_value = 0
				
				min_action, max_action = get_action_bounds(row, col)
				
				# Iterate through all possible actions from this state
				for action in range(min_action, max_action+1):
					
					V = get_state_value(row, col, action, value)
					
					if V > best_value:
						best_value = V
						best_action = action
				
				if best_action != curr_action:
					policy_change_count += 1
					policy[row,col] = best_action
					not_stable = True
							
		unstable_count += 1
		
		
	fig, ax = plt.subplots()
	cax = ax.imshow(policy)
	cbar = fig.colorbar(cax)
	plt.show()
		
	
		
if __name__=="__main__":
	main()
