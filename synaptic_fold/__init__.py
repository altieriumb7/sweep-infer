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
from .synaptic_cortical_column_7b import SynapticCorticalColumn7B, CorticalColumn7BConfig
from .cortical_adapter import CorticalManifoldAdapter
from .spatial_router import NeurosynapticSparseMLP, SpatialSynapticRouter
from .synaptic_dual_brain import SynapticDualBrain
from .converter import NeurosynapticConverter, NeurosynapticConfig
from .cortical_column_engine import CorticalMinicolumnMLP
from .native_engine import SynapticNativeEngine, SynapticResponse
from .cortical_gguf_pruner import prune_gguf_cortical_columns
from .kernels import is_native_avx2_available, compute_column_potentials_avx2
from .native_synaptic import (
    NativeSynapticMLP,
)
from .structural_synaptic import (
    StructuralSynapticMLP,
    StructuralSynapticCompressor,
)
from .synaptic_native_model import (
    SynapticNativeConfig,
    SynapticNativeModel,
)
from .synaptic_full28_model import (
    SynapticInt4Linear,
    SynapticFull28Model,
)
from .neuromorphic_synapse import (
    SynapticColumnWeight,
    NeuromorphicSynapticMLP,
    SynapseSerializer,
    convert_layer_to_synapse,
    factorize_weight_matrix,
)
from .synaptic_cortical_engine import (
    FusedSynapticLinear,
    FusedSynapticAttention,
    FusedSynapticMLP,
    SynapticInPlaceKVCache,
    SynapticCorticalEngine,
)
from .synaptic_anytime_cortical_engine import (
    PlattCalibrator,
    SynapticAnytimeCorticalEngine,
)
from .tree_speculator import (
    TreeNode,
    TreeTopology,
    SyntacticGrammarPrior,
    DomainSemanticCache,
    EntropyMarginAnalyzer,
    EntropyGuidedTreeSpeculator,
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
    "NativeSynapticMLP",
    "StructuralSynapticMLP",
    "StructuralSynapticCompressor",
    "SynapticNativeConfig",
    "SynapticNativeModel",
    "SynapticInt4Linear",
    "SynapticFull28Model",
    "SynapticColumnWeight",
    "NeuromorphicSynapticMLP",
    "SynapseSerializer",
    "convert_layer_to_synapse",
    "factorize_weight_matrix",
    "FusedSynapticLinear",
    "FusedSynapticAttention",
    "FusedSynapticMLP",
    "SynapticInPlaceKVCache",
    "SynapticCorticalEngine",
    "PlattCalibrator",
    "SynapticAnytimeCorticalEngine",
    "SynapticCorticalColumn7B",
    "CorticalColumn7BConfig",
    "CorticalManifoldAdapter",
    "SynapticDualBrain",
    "NeurosynapticSparseMLP",
    "SpatialSynapticRouter",
    "NeurosynapticConverter",
    "NeurosynapticConfig",
    "CorticalMinicolumnMLP",
    "SynapticNativeEngine",
    "SynapticResponse",
    "TreeNode",
    "TreeTopology",
    "SyntacticGrammarPrior",
    "DomainSemanticCache",
    "EntropyMarginAnalyzer",
    "EntropyGuidedTreeSpeculator",
]
