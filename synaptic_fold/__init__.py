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
from .intrinsic import (
    kneedle_knee_point,
    donoho_johnstone_threshold,
    IntrinsicSynapticMLPHook,
    IntrinsicModelWrapper,
)
from .quantized_synaptic import (
    Int4BlockQuantizer,
    Int4SynapticLinear,
    Int4SynapticMLP,
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
    "kneedle_knee_point",
    "donoho_johnstone_threshold",
    "IntrinsicSynapticMLPHook",
    "IntrinsicModelWrapper",
    "Int4BlockQuantizer",
    "Int4SynapticLinear",
    "Int4SynapticMLP",
]
