class RLGame(object):
    def __init__(self, world, agents):
        self.world = world
        self.agents = agents

    def RunEpisode(self):

        A = []
        S = [agent.curr_state]
        R = []
        n = 1

        for agent in agents:
            while not self._IsTerminal(agent):
                A.push_back( agent.GetAction( S[-1] ) )
                S.push_back( world.GetNextState( S[-1] , A[-1] ) )
                R.push_back( world.GetReward( S[-1] ) )

                if len(A) > n+1 and len(R) > n+1 and len(S) > n+1:
                    packet = ExpPacket(S[-2:], A[-2], R[-1], n)

                new_val = agent.ImprovePolicy(packet)
                agent.UpdateCurrentState( S[-1] )

                del packet

        return ExpPacket(S, A, R, n)

    def TrainAgent(self, id, episodes):
        pass

    def _IsTerminal(self, agent):
        return agent.curr_state == self.world.IsTerminal(agent.curr_state)
