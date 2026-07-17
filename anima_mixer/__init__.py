"""Anima Artist Mixer - multi-artist style mixing for the Anima model.

Hooks Anima/MiniTrainDIT cross-attention layers to mix independently encoded
artist conditionings, sidestepping LLM text-encoder interference when several
artist tags share one prompt. See CHANGELOG.md for version history.
"""

from .nodes_core import (
    AnimaArtistBasic,
    AnimaArtistCrossAttn,
    AnimaArtistPack,
    AnimaArtistPresetApply,
    AnimaArtistProbe,
    AnimaArtistProbeReport,
)
from .nodes_diagnostics import (
    AnimaArtistABVariants,
    AnimaArtistImpactMap,
    AnimaArtistTagCheck,
)
from .nodes_ui import (
    AnimaArtistChainBuilder,
    AnimaArtistChainPreview,
    AnimaArtistInspector,
    AnimaArtistOptions,
    AnimaArtistPreset,
    AnimaArtistRecipeLoad,
    AnimaArtistRecipeSave,
    AnimaArtistSimpleOptions,
    AnimaArtistStarter,
    AnimaArtistStyleBalance,
)

NODE_CLASS_MAPPINGS = {
    "AnimaArtistBasic": AnimaArtistBasic,
    "AnimaArtistStarter": AnimaArtistStarter,
    "AnimaArtistChainBuilder": AnimaArtistChainBuilder,
    "AnimaArtistChainPreview": AnimaArtistChainPreview,
    "AnimaArtistSimpleOptions": AnimaArtistSimpleOptions,
    "AnimaArtistPack": AnimaArtistPack,
    "AnimaArtistPresetApply": AnimaArtistPresetApply,
    "AnimaArtistCrossAttn": AnimaArtistCrossAttn,
    "AnimaArtistOptions": AnimaArtistOptions,
    "AnimaArtistStyleBalance": AnimaArtistStyleBalance,
    "AnimaArtistPreset": AnimaArtistPreset,
    "AnimaArtistInspector": AnimaArtistInspector,
    "AnimaArtistRecipeSave": AnimaArtistRecipeSave,
    "AnimaArtistRecipeLoad": AnimaArtistRecipeLoad,
    "AnimaArtistProbe": AnimaArtistProbe,
    "AnimaArtistProbeReport": AnimaArtistProbeReport,
    "AnimaArtistTagCheck": AnimaArtistTagCheck,
    "AnimaArtistABVariants": AnimaArtistABVariants,
    "AnimaArtistImpactMap": AnimaArtistImpactMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistBasic": "Anima Artist Basic (Recommended)",
    "AnimaArtistStarter": "Anima Artist Starter",
    "AnimaArtistChainBuilder": "Anima Artist Chain Builder",
    "AnimaArtistChainPreview": "Anima Artist Chain Preview",
    "AnimaArtistSimpleOptions": "Anima Artist Options (Simple)",
    "AnimaArtistPack": "Anima Artist Pack (Split + Encode)",
    "AnimaArtistPresetApply": "Anima Artist Apply Preset",
    "AnimaArtistCrossAttn": "Anima Artist Cross-Attn (Manual/Advanced)",
    "AnimaArtistOptions": "Anima Artist Options (Expert)",
    "AnimaArtistStyleBalance": "Anima Artist Style Balance",
    "AnimaArtistPreset": "Anima Artist Preset (Advanced)",
    "AnimaArtistInspector": "Anima Artist Inspector",
    "AnimaArtistRecipeSave": "Anima Artist Recipe (Save)",
    "AnimaArtistRecipeLoad": "Anima Artist Recipe (Load)",
    "AnimaArtistProbe": "Anima Artist Layer Probe",
    "AnimaArtistProbeReport": "Anima Artist Probe Report",
    "AnimaArtistTagCheck": "Anima Artist Tag Check (Encoder)",
    "AnimaArtistABVariants": "Anima Artist A/B Variants",
    "AnimaArtistImpactMap": "Anima Artist Impact Map (A/B Diff)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
