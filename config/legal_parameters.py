# Add agents, environments and rewards here to consider them legal options

AGENT_LIST = ["DoNothing",
              "Random",
              "DuelQSimple",
              "DQNPER",
              "DoublePER"]

NO_TRAINING_AGENTS_LIST = ["DoNothing",
                           "Random"]

ENV_LIST = ["l2rpn_case14_sandbox",
            "rte_case14_realistic",
            "l2rpn_wcci_2020",
            "l2rpn_neurips_2020_track1_small",
            "l2rpn_neurips_2020_track2_small"]

REWARD_LIST = ["L2RPNReward",
               "EpisodeDurationReward",
               "DistanceReward",
               "EconomicReward",
               "CloseToOverflowReward",
               "BridgeReward",
               "LinesCapacityReward",
               "LinesReconnectedReward",
               "CombinedScaledReward",
               "CapacityReward",
               "CapacityLineReconnectReward"]

NPZ_LIST = ['actions',
            'agent_exec_times',
            'disc_lines_cascading_failure',
            'env_modifications',
            'observations',
            'opponent_attack',
            'rewards']
