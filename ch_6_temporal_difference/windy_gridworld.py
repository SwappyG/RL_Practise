import numpy as np
import matplotlib.pyplot as plt
from Gridworld import Gridworld
from GridAgent import GridAgent

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

EPS = 0.1
ALPHA = 0.5
ALPHA_RAMP = 1
GAMMA = 1

PRINT_AFTER_MOVES = 1000
PRINT_AFTER_EPISODES = 100
TRAIN_EPS = 500
MOVE_TIMEOUT = 2000



def main():

	print(f"Starting Windy Gridworld, initializing world and agent")

	windy_world_args = {	"H": H, 
							"W": W, 
							"wind": WIND, 
							"start_pos": START_POS, 
							"end_pos": END_POS,
							"stochastic": True,
							"variance": 1				}
	
	windy_agent_args = {	"alpha": ALPHA,
							"eps": EPS,
							"gamma": GAMMA,
							"alpha_ramp": ALPHA_RAMP,
							"actions": ACTIONS 			}
	
	windy_world = Gridworld(**windy_world_args)
	windy_agent = GridAgent(START_POS, windy_world.H, windy_world.W, NUM_A, **windy_agent_args)
	# windy_agent.set_Q_to_default()
	
	print(f"Training agent for {TRAIN_EPS} episodes")
	windy_agent.train_agent(windy_world, print_moves=PRINT_AFTER_MOVES, move_timeout=MOVE_TIMEOUT, episodes=TRAIN_EPS)
		
	print(f"\nDone training agent for {TRAIN_EPS} episodes")
	print(f"Checking path and returns for trained agent")
	G, path = windy_agent.run_episode(windy_world, train=False)
	
	print(f"GridAgent completed task in {-G} moves\n")
	policy = windy_agent.get_policy(world=windy_world, visual=True)
	for a in policy:
		print(a)
	print()
	
	print(windy_world.get_image())
	
	y,x = zip(*path)
	fig, ax = plt.subplots(2,1)
	ax[0].imshow( windy_world.get_image(), origin="upper" )
	ax[0].plot(x,y)
	ax[1].plot(windy_agent.G_list)
	plt.show()
	
if __name__=="__main__":
	main()