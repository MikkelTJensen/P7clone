def get_agent_class(agent_name: str):
    if agent_name == "DQNPER":
        from agents.DQNPER import DQNPER
        return DQNPER
    if agent_name == "DoublePER":
        from agents.DoublePER import DoublePER
        return DoublePER


def get_reward_class(reward_name: str):
    if reward_name == "L2RPNReward":
        from grid2op.Reward import L2RPNReward
        return L2RPNReward
    elif reward_name == "EpisodeDurationReward":
        from grid2op.Reward import EpisodeDurationReward
        return EpisodeDurationReward
    elif reward_name == "DistanceReward":
        from grid2op.Reward import DistanceReward
        return DistanceReward
    elif reward_name == "EconomicReward":
        from grid2op.Reward import EconomicReward
        return EconomicReward
    elif reward_name == "LinesCapacityReward":
        from grid2op.Reward import LinesCapacityReward
        return LinesCapacityReward
    elif reward_name == "CombinedScaledReward":
        from grid2op.Reward import CombinedScaledReward
        return CombinedScaledReward
    elif reward_name == "CapacityReward":
        from rewards.CapacityReward import CapacityReward
        return CapacityReward
    elif reward_name == "CapacityLineReconnectReward":
        from rewards.CapacityLineReconnectReward import CapacityLineReconnectReward
        return CapacityLineReconnectReward

def get_combined_rewards():
    from grid2op.Reward import EpisodeDurationReward, LinesCapacityReward, CloseToOverflowReward, BridgeReward, LinesReconnectedReward
    return EpisodeDurationReward, LinesCapacityReward, CloseToOverflowReward, BridgeReward, LinesReconnectedReward