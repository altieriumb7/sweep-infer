"""
SynapticFold v2: Zero-Shot Recurrent Cortical Folding & Neuromorphic Execution.
"""

from .spiking import (
    SpikeIntegrateAndFire,
    TraceSwiGLU,
    WOSSAttention,
    OptimalBrainSpiking,
    fold_rmsnorm_affine,
)
from .folding import (
    compute_linear_cka,
    orthogonal_procrustes_alignment,
    regmean_merge,
    RecurrentCorticalFolder,
)
from .lyapunov import (
    estimate_spectral_properties,
    calibrate_lyapunov_damping,
    KrasnoselskiiMannStep,
)
from .consensus import (
    AdaptiveConsensusCheck,
    ConsensusMetrics,
)
from .reservoir import (
    GlobalSynapticPool,
    ContinuousAttractorDynamics,
    ReservoirConfig,
)
from .engine import (
    SynapticBrainConfig,
    SynapticBrainEngine,
    GenerationOutput,
    InPlaceReservoirCache,
)
from .adapter import (
    UniversalModelAdapter,
)

__version__ = "2.0.0"

__all__ = [
    "SpikeIntegrateAndFire",
    "TraceSwiGLU",
    "WOSSAttention",
    "OptimalBrainSpiking",
    "fold_rmsnorm_affine",
    "compute_linear_cka",
    "orthogonal_procrustes_alignment",
    "regmean_merge",
    "RecurrentCorticalFolder",
    "estimate_spectral_properties",
    "calibrate_lyapunov_damping",
    "KrasnoselskiiMannStep",
    "AdaptiveConsensusCheck",
    "ConsensusMetrics",
    "GlobalSynapticPool",
    "ContinuousAttractorDynamics",
    "ReservoirConfig",
    "SynapticBrainConfig",
    "SynapticBrainEngine",
    "GenerationOutput",
    "InPlaceReservoirCache",
    "UniversalModelAdapter",
]
