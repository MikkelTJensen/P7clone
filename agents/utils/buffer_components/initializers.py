def get_sampler_class(sampler: str):
    if sampler == 'simple':
        from agents.utils.buffer_components.samplers.simple_sampler import SimpleSampler
        return SimpleSampler
    elif sampler == 'tdreward':
        from agents.utils.buffer_components.samplers.tdreward_sampler import TDRewardSampler
        return TDRewardSampler
    elif sampler == 'sequence':
        from agents.utils.buffer_components.samplers.sequence_sampler import SequenceSampler
        return SequenceSampler
    elif sampler == 'floating':
        from agents.utils.buffer_components.samplers.average_sampler import AverageSequenceSampler
        return AverageSequenceSampler
    elif sampler == 'staleness':
        from agents.utils.buffer_components.samplers.staleness_sampler import StalenessSampler
        return StalenessSampler
    elif sampler == 'blackout':
        from agents.utils.buffer_components.samplers.blackout_sampler import BlackoutSampler
        return BlackoutSampler
    elif sampler == 'fitty_blackout':
        from agents.utils.buffer_components.samplers.fitty_blackout_sampler import FittyBlackoutSampler
        return FittyBlackoutSampler


def get_prioritizer_class(prioritizer: str):
    if prioritizer == 'simple':
        from agents.utils.buffer_components.prioritizers.simple_prioritizer import SimplePrioritizer
        return SimplePrioritizer


def get_deprioritizer_class(deprioritizer: str):
    if deprioritizer == 'simple':
        from agents.utils.buffer_components.deprioritizers.simple_deprioritizer import SimpleDeprioritizer
        return SimpleDeprioritizer
    elif deprioritizer == 'sequence':
        from agents.utils.buffer_components.deprioritizers.sequence_deprioritizer import SequenceDeprioritizer
        return SequenceDeprioritizer
    elif deprioritizer == 'staleness':
        from agents.utils.buffer_components.deprioritizers.staleness_deprioritizer import StalenessDeprioritizer
        return StalenessDeprioritizer


def get_experience_class(experience: str):
    if experience == 'simple':
        from agents.utils.buffer_components.experiences.exp_importance import ImportanceExperience
        return ImportanceExperience
    elif experience == 'sequence':
        from agents.utils.buffer_components.experiences.exp_sequence import SequenceExperience
        return SequenceExperience
    elif experience == 'staleness':
        from agents.utils.buffer_components.experiences.exp_staleness import StalenessExperience
        return StalenessExperience
