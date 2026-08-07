from profiler_study.modes import ProfilerEnvironment
from profiler_study.analyzer import ProfilerAnalyzer

def get_oracle_env(base: float = 120.0):
    return ProfilerEnvironment(base)

def get_oracle_analyzer(env):
    return ProfilerAnalyzer(env)
