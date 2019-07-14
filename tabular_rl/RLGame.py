class RLGame(object):
    def __init__(self, world, agents):
        self.world = world
        self.agents = agents

    def RunEpisode(self):

        agent_hists = []
        for agent in self.agents:
            agent_hist.push_back( ExpPacket([], [agent.curr_state], [], 1) )

        for ii in range(len(self.agents)):
            while not self._IsTerminal(self.agents[ii]):
                self.StepAgent(self.agents[ii], agent_hists[ii])

        return ExpPacket(S, A, R, n)

    def StepAgent(self, agent, agent_hist):
        A = agent.GetAction( S[-1] )
        S = world.GetNextState( S[-1] , A[-1] )
        R = world.GetReward( S[-1] )

        agent_hist.Push(S, A, R)

        if agent_hist.Trainable():
            S, A, R, _ = agent_hist.Get()
            packet = ExpPacket(S[-2:], A[-2], R[-1], n)

        new_val = agent.ImprovePolicy(packet)
        agent.UpdateCurrentState( S[-1] )

        del packet

    def TrainAgent(self, id, episodes):
        pass

    def _IsTerminal(self, agent):
        return agent.curr_state == self.world.IsTerminal(agent.curr_state)
