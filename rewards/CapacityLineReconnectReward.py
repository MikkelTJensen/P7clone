from grid2op.Reward.BaseReward import BaseReward
import numpy as np
from grid2op.dtypes import dt_float

class CapacityLineReconnectReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)
    

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # returns min reward if there is a game over
        if has_error:
            return self.reward_min
        
        penalty = dt_float(0.1)

        minimum_border = self.reward_min + penalty
        
        #get the obs
        obs = env.get_obs()

        all_lines = np.arange(env.n_line)

        total_usage = 0
        
        usage = obs.rho[obs.line_status == True]

        for use in usage:
            total_usage = total_usage + use

        percentage_used = total_usage / len(usage)

        reward = self.reward_max - percentage_used
        
        no_cooldown = all_lines[obs.time_before_cooldown_line == 0]

        for line in no_cooldown:
            if obs.line_status[line] == False and reward > minimum_border:
                reward = reward - penalty

        return dt_float(reward)
