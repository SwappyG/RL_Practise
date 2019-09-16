from Agent import Agent
from TabularPolicy import TabularPolicy

class TabularAgent(Agent):

	def __init__(self, policy, start_state, **kwargs):

		if not isinstance(policy, TabularPolicy):
			raise TypeError("policy argument must be of type TabularPolicy")

		Agent.__init__(self, policy, start_state, **kwargs)