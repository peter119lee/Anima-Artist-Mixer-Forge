"""Shared constants for the Anima Artist Mixer node pack."""

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"
FUSION_BASE_PRESERVE = "base_preserve"

FUSION_CHOICES = [
    FUSION_INTERPOLATE,
    FUSION_CONCAT_WITH_BASE,
    FUSION_BASE_PRESERVE,
]

COMBINE_CONCAT = "concat"
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_LOWRANK_AVG = "lowrank_avg"

COMBINE_CHOICES = [
    COMBINE_CONCAT,
    COMBINE_OUTPUT_AVG,
    COMBINE_LOWRANK_AVG,
]

MAX_ARTISTS = 32

# Linear injection weight range (supports negative weights for style subtraction).
WEIGHT_MIN = -4.0
WEIGHT_MAX = 4.0

STATIC_CAPTURE_K_DEFAULT = 6   # default step count for the H' temporal average
STATIC_CAPTURE_K_MAX = 12      # UI upper bound
STATIC_CAPTURE_MODE_OUTPUT = "output"
STATIC_CAPTURE_MODE_DELTA = "delta"
STATIC_CAPTURE_MODE_BLEND = "blend"
STATIC_CAPTURE_MODE_BLEND_PERP = "blend_perp"
STATIC_CAPTURE_BLEND_ALPHA_DEFAULT = 0.25

STATIC_CAPTURE_MODE_CHOICES = [
    STATIC_CAPTURE_MODE_OUTPUT,
    STATIC_CAPTURE_MODE_DELTA,
    STATIC_CAPTURE_MODE_BLEND,
    STATIC_CAPTURE_MODE_BLEND_PERP,
]

NORM_LOCK_ROW = "row"
NORM_LOCK_TOKEN = "token"

NORM_LOCK_SCOPE_MIXED = "mixed"
NORM_LOCK_SCOPE_PER_ARTIST = "per_artist"
NORM_LOCK_SCOPE_BOTH = "both"

CONTRIB_BALANCE_ALPHA_DEFAULT = 1.0
CONTRIB_BALANCE_MIN_SCALE = 0.05
CONTRIB_BALANCE_MAX_SCALE = 4.0

MIXED_DELTA_CAP_RATIO_DEFAULT = 1.0
MIXED_DELTA_CAP_RATIO_MAX = 4.0
DRIFT_AUTO_CLOSEUP_DELTA_CAP_RATIO = 0.75

ANCHOR_SEED = 42                          # default single-anchor seed
ANCHOR_SEEDS_POOL = [42, 100, 200, 300]   # seeds used when averaging multiple anchors
ANCHOR_SEEDS_MAX = 4                      # UI upper bound = len(ANCHOR_SEEDS_POOL)
ANCHOR_LAYER_THRESHOLD_DISABLED = -1      # -1 means every layer uses the anchor Q

# v26: Stabilizer end percent - allows stabilizers to auto-disable after certain sampling progress
# for compatibility with FLS (Foveated Latent Sampler) and similar step-delta-based samplers
STABILIZER_END_PERCENT_DEFAULT = 1.0      # 1.0 = full sampling (v24 behavior, backward compatible)
STABILIZER_END_PERCENT_FLS_RECOMMENDED = 0.5  # 0.5 = stabilizer active first 50%, dynamic after

PRESET_BALANCED = "balanced"
PRESET_STRONG_STYLE = "strong_style"
PRESET_STABLE_SEED = "stable_seed"
PRESET_DRIFT_AUTO = "drift_auto"
PRESET_DRIFT_SOFT = "drift_soft"
PRESET_FACE_LOCK = "face_lock"
PRESET_SCENE_LOCK = "scene_lock"
PRESET_ANCHOR_LOCK = "anchor_lock"
PRESET_FAST_PREVIEW = "fast_preview"
PRESET_IDENTITY_GUARD = "identity_guard"
PRESET_COMPATIBILITY_SAFE = "compatibility_safe"
PRESET_COMPATIBILITY_SAFE_9_15 = "compatibility_safe_9_15"

PRESET_CHOICES = [
    PRESET_BALANCED,
    PRESET_STRONG_STYLE,
    PRESET_STABLE_SEED,
    PRESET_DRIFT_AUTO,
    PRESET_DRIFT_SOFT,
    PRESET_FACE_LOCK,
    PRESET_SCENE_LOCK,
    PRESET_ANCHOR_LOCK,
    PRESET_FAST_PREVIEW,
    PRESET_IDENTITY_GUARD,
    PRESET_COMPATIBILITY_SAFE,
]

PRESET_RECOMMENDED_CHOICES = [
    PRESET_BALANCED,
    PRESET_STRONG_STYLE,
    PRESET_DRIFT_AUTO,
]

LAYER_MODE_AUTO = "auto"
LAYER_MODE_ALL = "all_layers"
LAYER_MODE_STYLE_CORE = "style_core"
LAYER_MODE_DETAIL = "detail_layers"
LAYER_MODE_CUSTOM = "custom"

LAYER_MODE_CHOICES = [
    LAYER_MODE_AUTO,
    LAYER_MODE_ALL,
    LAYER_MODE_STYLE_CORE,
    LAYER_MODE_DETAIL,
    LAYER_MODE_CUSTOM,
]

CHAIN_LAYOUT_MANUAL = "manual"
CHAIN_LAYOUT_EVEN_LAYERS = "even_layers"
CHAIN_LAYOUT_LAYER_SCHEDULED = "layer_scheduled"

CHAIN_LAYOUT_CHOICES = [
    CHAIN_LAYOUT_MANUAL,
    CHAIN_LAYOUT_EVEN_LAYERS,
    CHAIN_LAYOUT_LAYER_SCHEDULED,
]

DEFAULT_NUM_BLOCKS = 28  # Anima/MiniTrainDIT default DiT block count

RECIPE_FORMAT = "anima-artist-recipe"
RECIPE_VERSION = 1
