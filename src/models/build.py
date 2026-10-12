# =============================================================================
# BUILD: CREATE A DEEPONET OR FNO FROM THE SETTINGS, OR FROM A CHECKPOINT
#
# Architecture fields live in cfg (see src/config.py). A named variant is
# only a shortcut that fills those fields:
#     cfg.variant = "baseline"  ->  the architecture of the published study
#     cfg.variant = "custom"    ->  use the cfg fields as they are
#                                   (this is what the tuning grids use)
# Every checkpoint stores the architecture dictionary, so a model is always
# rebuilt from the checkpoint itself, never from the current settings.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   n_points     : grid points per case, Ny*Nx (DeepONet branch input size)
#   ny, nx       : grid size (FNO)
#   architecture : dictionary stored in the checkpoint (see deeponet.py/fno.py)
# =============================================================================

from .deeponet import DeepONet
from .fno import FNO2d


# =============================================================================
# NAMED ARCHITECTURES (study of the original report; one change per variant)
# =============================================================================
DEEPONET_VARIANTS = {
    "baseline":        {"branch_hidden": (256, 256), "trunk_hidden": (128, 128),
                        "latent_dim": 128, "deeponet_activation": "tanh"},
    "activation_gelu": {"branch_hidden": (256, 256), "trunk_hidden": (128, 128),
                        "latent_dim": 128, "deeponet_activation": "gelu"},
    "depth3":          {"branch_hidden": (256, 256, 256), "trunk_hidden": (128, 128, 128),
                        "latent_dim": 128, "deeponet_activation": "tanh"},
    "latent256":       {"branch_hidden": (256, 256), "trunk_hidden": (128, 128),
                        "latent_dim": 256, "deeponet_activation": "tanh"},
    "depth4_gelu":     {"branch_hidden": (256,) * 4, "trunk_hidden": (128,) * 4,
                        "latent_dim": 128, "deeponet_activation": "gelu"},
    "depth6_gelu":     {"branch_hidden": (256,) * 6, "trunk_hidden": (128,) * 6,
                        "latent_dim": 128, "deeponet_activation": "gelu"},
}

FNO_VARIANTS = {
    "baseline":        {"fno_blocks": 4, "fno_activation": "gelu", "fno_modes_y": 12, "fno_modes_x": 16},
    "activation_tanh": {"fno_blocks": 4, "fno_activation": "tanh", "fno_modes_y": 12, "fno_modes_x": 16},
    "depth3":          {"fno_blocks": 3, "fno_activation": "gelu", "fno_modes_y": 12, "fno_modes_x": 16},
    "modes8":          {"fno_blocks": 4, "fno_activation": "gelu", "fno_modes_y": 8, "fno_modes_x": 8},
}


def available_variants(operator):
    if operator == "deeponet":
        return DEEPONET_VARIANTS
    if operator == "fno":
        return FNO_VARIANTS
    raise ValueError("Operator must be deeponet or fno")


# =============================================================================
# COPY A NAMED VARIANT INTO THE SETTINGS ("custom" leaves cfg unchanged)
# =============================================================================
def apply_variant(cfg):
    if cfg.variant == "custom":
        return cfg

    variants = available_variants(cfg.operator)
    if cfg.variant not in variants:
        raise ValueError("Unknown configuration: " + cfg.variant)

    fields = variants[cfg.variant]
    for name in fields:
        setattr(cfg, name, fields[name])
    return cfg


# =============================================================================
# NEW MODEL FROM THE SETTINGS
# =============================================================================
def build_deeponet(cfg, n_points):
    return DeepONet(
        branch_in=n_points,
        branch_hidden=list(cfg.branch_hidden),
        trunk_hidden=list(cfg.trunk_hidden),
        latent=cfg.latent_dim,
        activation=cfg.deeponet_activation,
    )


def build_fno(cfg, ny, nx):
    # MODES CANNOT EXCEED WHAT THE PADDED GRID HOLDS (matters on tiny test grids only)
    modes_y = min(cfg.fno_modes_y, (ny + cfg.fno_padding) // 2)
    modes_x = min(cfg.fno_modes_x, (nx + cfg.fno_padding) // 2 + 1)

    settings = {
        "blocks": cfg.fno_blocks,
        "activation": cfg.fno_activation,
        "modes_y": modes_y,
        "modes_x": modes_x,
        "input_channels": 3,
        "output_channels": 1,
        "width": cfg.fno_width,
        "padding": cfg.fno_padding,
    }
    model = FNO2d(**settings)

    model.model_kind = "fno"
    model.architecture = dict(settings)
    model.architecture["grid_ny"] = ny
    model.architecture["grid_nx"] = nx
    return model


def build_model(cfg, ny, nx):
    if cfg.operator == "deeponet":
        return build_deeponet(cfg, ny * nx)
    if cfg.operator == "fno":
        return build_fno(cfg, ny, nx)
    raise ValueError("Operator must be deeponet or fno")


# =============================================================================
# MODEL FROM A SAVED CHECKPOINT (weights are loaded by the caller)
# =============================================================================
def model_from_checkpoint(checkpoint):
    settings = dict(checkpoint["architecture"])

    if checkpoint["model_kind"] == "deeponet":
        return DeepONet(**settings)

    if checkpoint["model_kind"] == "fno":
        ny = settings.pop("grid_ny")
        nx = settings.pop("grid_nx")
        model = FNO2d(**settings)
        model.model_kind = "fno"
        model.architecture = dict(settings)
        model.architecture["grid_ny"] = ny
        model.architecture["grid_nx"] = nx
        return model

    raise ValueError("Checkpoint holds an unsupported model: %s" % checkpoint["model_kind"])


# =============================================================================
# NUMBER OF REAL TRAINABLE SCALARS (a complex weight counts as two)
# =============================================================================
def parameter_count(model):
    total = 0
    for parameter in model.parameters():
        if parameter.is_complex():
            total = total + 2 * parameter.numel()
        else:
            total = total + parameter.numel()
    return total
