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
        for agent in self._agents:
            self._history[agent.GetID()] = ExpPacket()

        self._episodes = []

    def GetLatestEpisodeHistory(self):
        if len(self._episodes) == 0:
            WARN("No episodes have been run on this game")
            return None
        
        return copy.deepcopy(self._episodes[-1])


    def GetAllEpisodes(self):
        return self._episodes

    def RunEpisode(self, max_steps = RLGame.DEFAULT_NUM_STEPS_PER_EP):

        self._history.clear()

        # create flag to keep track of whether all agents are terminal
        all_agents_terminal = True
        step = 0

        while step < max_steps:
            # Iterate through all agents in the game
            for agent in self._agents:

                # if the current agent is terminal, move to next agent
                if self._IsTerminal(agent):
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

    def _StepAgent(self, agent):

        # grab the current state
        s_prev = agent.GetCurrState()

        # Take action and get next state and reward
        a_next = agent.GetAction(s_prev)
        s_next = self._world.GetNextState(s_prev, a_next)
        r_next = self._world.GetReward(s_next)

        # Update the state of the agent and store the S A R triplet into the history
        agent.UpdateCurrentState( s_next )
        self._history[agent.GetID()].Push(s_next, a_next, r_next)

        # If this is a trainable agent, then attempt to train
        if agent.Trainable():
            s_req, a_req, r_req = agent.PacketSizeReq()
            packet = self._history[agent.GetID()].GetLatestAsPacket(s_req, a_req, r_req)
            if packet != None:
                if not agent.ImprovePolicy(packet):
                    WARN("Failed to train agent with ExpPacket {packet}")

        # Delete the packet which we don't need anymore
        del packet

    def StepAgentByID(self, id):
        try:
            agent = self._agents[id]
        except KeyError:
            ERROR(f"Invalid key passed to StepAgent, {id} is not a valid agent id")
            return False

        self._StepAgent(agent)

    def TrainAgent(self, id, episodes = RLGame.DEFAULT_NUM_EPS, steps_per_episode = RLGame.DEFAULT_NUM_STEPS_PER_EP):
        try:
            agent_to_train = self._agents[id]
        
        except KeyError:
            ERROR(f"Invalid key passed to StepAgent, {id} is not a valid agent id")
            return False

        for agent in self._agents:
            if agent is agent_to_train:
                agent.SetTrainable(True)
            else:
                agent.SetTrainable(False)

        for ii in range(episodes):
            INFO(f"Running episode {ii}")
            steps, _ = self.RunEpisode(steps_per_episode)
            INFO(f"Ran for {steps} steps")

    def _IsTerminal(self, agent):
        return self._world.IsTerminal(agent.GetCurrState())
        
