import numpy as np
from scipy import stats
from collections import deque
import matplotlib.pyplot as plt
from windy_gridworld_class import Gridworld, Agent

# Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
UP_LEFT = 4
UP_RIGHT = 5
DOWN_LEFT = 6
DOWN_RIGHT = 7
ACTIONS = (UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)

NUM_A = 8
H = 7
W = 10

START_POS = (3, 0)
END_POS = (3, 7)

WIND = (0, 0, 0, -1, -1, -1, -2, -2, -1, 0)

GAMMA = 1

PRINT_AFTER_MOVES = 1000
PRINT_AFTER_EPISODES = 100
TRAIN_EPS = 1000
MOVE_TIMEOUT = 2000

EPS = 0.1
ALPHA = 0.5

def main():

	print(f"Starting Windy Gridworld, initializing world and agent")

	world_args = {"H": H, "W": W, "wind": WIND, "start_pos":START_POS, "end_pos":END_POS}
	world = Gridworld(**world_args)
	agent = Agent(START_POS, world.H, world.W, NUM_A, **{"actions": ACTIONS})
	
	print(f"Training agent for {TRAIN_EPS} episodes")
	agent.train_agent(world, print_moves=PRINT_AFTER_MOVES, move_timeout=MOVE_TIMEOUT, episodes=TRAIN_EPS)
		
	print(f"\nDone training agent for {TRAIN_EPS} episodes")
	print(f"Checking path and returns for trained agent")
	G, path = agent.run_episode(world, train=False)
	
	print(f"Agent completed task in {-G} moves\n")
	policy = agent.get_policy(world=world, visual=True)
	for a in policy:
		print(a)
	print()
	
	y,x = zip(*path)
	fig, ax = plt.subplots()
	ax.plot(x,y)
	ax.set_ylim(ax.get_ylim()[::-1])
	# ax[1].plot(G_list)
	plt.show()
	
if __name__=="__main__":
	main()