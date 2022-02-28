from grid2op.Reward.BaseReward import BaseReward
import numpy as np
from grid2op.dtypes import dt_float

class CapacityReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)
    

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # returns min reward if there is a game over
        if has_error:
            return self.reward_min
        
        #get the obs
        obs = env.get_obs()

        total_usage = 0
        
        usage = obs.rho[obs.line_status == True]

        for use in usage:
            total_usage = total_usage + use

        percentage_used = total_usage / len(usage)

        reward = self.reward_max - percentage_used

        return dt_float(reward)
