import numpy as np
import matplotlib.pyplot as plt
from Lineworld import Lineworld
from LineAgent import LineAgent


EPS = 1
ALPHA = 0.1
GAMMA = 1

L = 19
LEFT = 0
RIGHT = 1
ACTIONS = [LEFT, RIGHT]
 
START_POS = L//2
END_POS = (0, L-1)

TRAIN_EPS = 1000

def main():
	
	line_world_args = {
		"L": L,
		"start_pos": START_POS
	}
	
	line_agent_args = {
		"eps": EPS,
		"alpha": ALPHA,
		"gamma": GAMMA,
		"actions": ACTIONS
	}
	
	line_world = Lineworld(**line_world_args)
	line_agent = LineAgent(START_POS, line_world.L, **line_agent_args)
	
	print(f"Training agent for {TRAIN_EPS} episodes")
	line_agent.train_agent(
		line_world, 
		print_moves=100, 
		move_timeout=1000, 
		episodes=TRAIN_EPS,
		ramp_alpha=True, 
		method=LineAgent.SARSA,
		n_step=1
	)
	
	# Check the final policy (no training, acting 100% greedily)
	print(f"\nDone training agent for {TRAIN_EPS} episodes")
	print(f"Checking path and returns for trained agent")
	G, path = line_agent.run_episode(line_world, train=False)
	
	print(f"LineAgent received reward {G}	\n")
	
	# Plot the final path that agent took, and its rewards over course of training
	fig, ax = plt.subplots()
	ax.plot(line_agent.G_list)
	plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
if __name__=="__main__":
	main()