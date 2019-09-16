# -*- coding: future_fstrings -*-
import sys
import numpy as np

import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

import copy
from ExpPacket import ExpPacket

class RLGame(object):

	DEFAULT_NUM_EPS = 1
	DEFAULT_NUM_STEPS_PER_EP = 100

	def __init__(self, world, agents):
		"""
		Initializes an RLGame object, used to run episodes and train agents

		Params:
		- world: [World] the world that this game operates in (must be subclass of type World)
		- agents: [list] all the agents (initialized) participating in this game (must be list of type Agent)

		"""
		self._world = world
		self._agents = {}

		# Create dictionary of agents
		for agent in agents:
			self._agents[agent.GetID()] = agent

		self._history = {}
		
		# Build dictionary for history
		for id, _ in self._agents.items():
			self._history[id] = ExpPacket()

		self._episodes = []

	def GetLatestEpisodeHistory(self):
		if len(self._episodes) == 0:
			WARN("No episodes have been run on this game")
			return None
		
		return copy.deepcopy(self._episodes[-1])

	def GetCurrentEpisodeHistory(self):
		return copy.deepcopy(self._history)

	def GetCurrentAgentHistory(self, agent_id):
		try:
			return self._history[agent_id]
		except KeyError:
			raise ValueError(f"ID [{id}] provided is invalid, this agent is not in this game")
		
		return None

	def GetAllEpisodes(self):
		return self._episodes

	def GetCurrStateByAgentID(self, id):
		try: 
			return self._agents[id].GetCurrState()
		except IndexError:
			raise ValueError(f"ID [{id}] provided is invalid, this agent is not in this game")
		
		return None

	def TrainAgent(self, id, episodes = None, steps_per_episode = None):
		if episodes == None:
			episodes = RLGame.DEFAULT_NUM_EPS

		if steps_per_episode == None:
			steps_per_episode = RLGame.DEFAULT_NUM_STEPS_PER_EP
		
		try:
			agent_to_train = self._agents[id]
		
		except KeyError:
			ERROR(f"Invalid key passed to StepAgent, {id} is not a valid agent id")
			return False

		for _, agent in self._agents.items():
			if agent is agent_to_train:
				agent.SetTrainable(True)
			else:
				agent.SetTrainable(False)

		for ii in range(episodes):
			INFO(f"Running episode {ii}")
			steps, _ = self.RunEpisode(steps_per_episode)
			INFO(f"Ran for {steps} steps")

		INFO("Training complete")

	def RunEpisode(self, max_steps = None):

		if max_steps == None:
			max_steps = RLGame.DEFAULT_NUM_STEPS_PER_EP

		self._history.clear()

		# create flag to keep track of whether all agents are terminal
		all_agents_terminal = True
		step = 0

		while step < max_steps:
			# Iterate through all agents in the game
			for id, agent in self._agents.items():

				# if the current agent is terminal, move to next agent
				if self._IsTerminal(agent):
					self._history[id].Push(agent.GetCurrState(), None, None)
					continue

				# Otherwise, call StepAgent on the agent
				self._StepAgent(agent)

				# Update all_agents_terminal flag
				all_agents_terminal = (all_agents_terminal and self._IsTerminal(agent))

			step += 1

			# if all agents are terminal, break early
			if all_agents_terminal:
				INFO(f"Ending episode early at step {step}, all agents are at terminal state")
				self._episodes.append(self._history)
				return (step, self.GetLatestEpisodeHistory())

		INFO(f"Ran for specified number of max steps, at least some agents are still not at terminal state")
		self._episodes.append(self._history)
		return (step, self.GetLatestEpisodeHistory())

	def StepAgentByID(self, id):
		try:
			agent = self._agents[id]
		except KeyError:
			ERROR(f"Invalid key passed to StepAgent, {id} is not a valid agent id")
			return False

		self._StepAgent(agent)

	def _StepAgent(self, agent):

		# grab the current state
		s_prev = agent.GetCurrState()

		# Take action and get next state and reward
		a_next = agent.GetAction(s_prev)
		s_next = self._world.GetNextState(s_prev, a_next)
		r_next = self._world.GetReward(s_next)

		# Update the state of the agent and store the S A R triplet into the history
		agent.UpdateCurrentState( s_next )
		self._history[agent.GetID()].Push(s_prev, a_next, r_next)

		# If this is a trainable agent, then attempt to train
		if agent.IsTrainable():
			DEBUG(f"Training agent {agent.GetID()}")
			s_req, a_req, r_req = agent.PacketSizeReq()
			packet = self._history[agent.GetID()].GetLatestAsPacket(s_req, a_req, r_req)
			if packet != None:
				if not agent.ImprovePolicy(packet):
					WARN("Failed to train agent with ExpPacket {packet}")

		# Delete the packet which we don't need anymore
		del packet

	def _IsTerminal(self, agent):
		return self._world.IsTerminal(agent.GetCurrState())
		


if __name__=="__main__":

	import unittest
	from collections import OrderedDict
	from WorldSpace import WorldSpace
	from DynamicNDWorld import DynamicNDWorld
	from ExpPacket import ExpPacket
	from TabularAgent import TabularAgent
	from SarsaPolicy import SarsaPolicy

	class TestAgent(unittest.TestCase):

		def setUp(self):
			self.ss = (7,9)

			self.a_map = OrderedDict()
			self.a_map['U'] = (0,1)
			self.a_map['D'] = (0,-1)
			self.a_map['R'] = (1,0)
			self.a_map['L'] = (-1,0)

			self.p_kw = {}
			self.p_kw['discount_factor'] = 1
			self.p_kw['exploration_factor'] = 1
			self.p_kw['is_static'] = False
			self.p_kw['init_variance'] = 0.01

			self.ws = WorldSpace(self.ss, self.a_map)
			self.policy = SarsaPolicy(self.ws, **self.p_kw)

			def hazards_func(state):
				haz_states = []
				haz_states.append((1,1))
				haz_states.append((2,2))
				haz_states.append((3,3))
				haz_states.append((1,4))
				haz_states.append((4,1))

				DEBUG(f"STATE: -- {state}")

				if tuple(state) in haz_states:
					return True
				else:
					return False

			self.world_kw = {}
			self.world_kw['start_state'] = (0,0)
			self.world_kw['goal_state'] = (4,4)
			self.world_kw['hazard_func'] = hazards_func

			self.world = DynamicNDWorld(self.ws, **self.world_kw)

			self.agent_kw = {}
			self.agent_kw["ID"] = 5
			self.agent_kw["is_training"] = True

			self.agent = TabularAgent(self.policy, (0,0), **self.agent_kw)
			self.rl_game = RLGame(self.world, [self.agent])

		def test_StepAgentByID(self):

			# grab the id and init state
			id = self.agent_kw["ID"]
			init_state = self.agent.GetCurrState()

			# step the agent a few times
			for _ in range(300):
				self.rl_game.StepAgentByID(id)
			
			# check that the history is updating properly	
			hist = self.rl_game.GetCurrentAgentHistory(id)
			print hist
			self.assertTrue(len(hist) == 300)

	unittest.main()