"""Multi-agent environment adapters for MaCA."""

from .maca_parallel_env import MaCAParallelEnv
from .sample_factory_env import SampleFactoryMaCAEnv

__all__ = ["MaCAParallelEnv", "SampleFactoryMaCAEnv"]
