import numpy as np
import matplotlib.pyplot as plt
from Gridworld import Gridworld
from Agent import Agent

# Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
UP_LEFT = 4
UP_RIGHT = 5
DOWN_LEFT = 6
DOWN_RIGHT = 7		
ACTIONS = ACTIONS = (UP, DOWN, LEFT, RIGHT)

NUM_A = 4
H = 4
W = 12

START_POS = (3, 0)
END_POS = (3, 11)

WIND = (0,)*W #(0,)*3 + (1,)*6 + (0,)*3

HAZARD_W_COORS = range(1,W-1) # SCCC...CCCS  # S=safe, C=cliff
HAZARD_H_COORS = (3,) * W
HAZARD = list(zip(HAZARD_H_COORS, HAZARD_W_COORS))

# for h in HAZARD:
	# print(h)

GAMMA = 1

PRINT_AFTER_MOVES = 1000
PRINT_AFTER_EPISODES = 100
TRAIN_EPS = 5000
MOVE_TIMEOUT = 2000

EPS = 0.1
ALPHA = 0.1
ALPHA_RAMP = 1



def main():
	
	print(f"Starting Cliff Walk, initializing world and agent")
	
	# Create args to use for instantiating a world and agent object
	cliff_world_args = {	
		"H": H, 
		"W": W, 
		"wind": WIND, 
		"start_pos": START_POS, 
		"end_pos": END_POS,
		"stochastic": False,
		"variance": 0,
		"hazard": HAZARD
	}
	
	cliff_agent_args = {
		"alpha": ALPHA,
		"eps": EPS,
		"gamma": GAMMA,
		"alpha_ramp": ALPHA_RAMP,
		"actions": ACTIONS
	}
	
	# Create the world and agent
	cliff_world = Gridworld(**cliff_world_args)
	cliff_agent = Agent(START_POS, cliff_world.H, cliff_world.W, NUM_A, **cliff_agent_args)
	
	# Train the agent for specified eps, occasionally printing to console
	print(f"Training agent for {TRAIN_EPS} episodes")
	cliff_agent.train_agent(
		cliff_world, 
		print_moves=1000, 
		move_timeout=1000, 
		episodes=5000,
		ramp_alpha=True, 
		method=Agent.Q_LEARNING
	)
		
	# Check the final policy (no training, acting 100% greedily)
	print(f"\nDone training agent for {TRAIN_EPS} episodes")
	print(f"Checking path and returns for trained agent")
	cliff_agent.Q = cliff_agent.best_Q
	G, path = cliff_agent.run_episode(cliff_world, train=False)
	
	print(f"Agent received reward {-G} (smaller is better)\n")
	
	# Visualize the final policy
	policy = cliff_agent.get_policy(world=cliff_world, visual=True)
	for a in policy:
		print(a)
	print()
	
	# Plot the final path that agent took, and its rewards over course of training
	y,x = zip(*path)
	fig, ax = plt.subplots(2,1)
	ax[0].imshow( cliff_world.get_image(), origin="upper" )
	# ax[0].set_ylim(ax[0].get_ylim()[::-1])
	ax[0].plot(x,y)
	
	
	
	
	ax[1].plot(cliff_agent.G_list)
	plt.show()
	
	
	
	
	
	
	
	
	
	
	
if __name__=="__main__":
	main()