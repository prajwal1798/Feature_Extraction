# =============================================================================
# TEST: DEEPONET AND FNO  (models/, training/runtime.py)
#   parameter counts of the published architectures
#   chunked DeepONet backward == full-batch gradient (the memory trick is exact)
# =============================================================================

import torch

from src.config import Settings
from src.models.build import apply_variant, build_model, parameter_count
from src.training.runtime import build_case_dependent_trunk_input, fno_field


def model_for(operator, variant):
    cfg = Settings(operator=operator, variant=variant)
    apply_variant(cfg)
    torch.manual_seed(0)
    return build_model(cfg, 96, 128)


def test_parameter_counts():
    assert parameter_count(model_for("deeponet", "baseline")) == 3278209
    assert parameter_count(model_for("fno", "baseline")) == 3152257
    assert parameter_count(model_for("fno", "modes8")) == 1055105


def test_output_shapes():
    deeponet = model_for("deeponet", "baseline")
    fno = model_for("fno", "baseline")
    sdf = torch.randn(2, 96 * 128)
    coordinates = torch.rand(96 * 128, 2)

    trunk_input = build_case_dependent_trunk_input(coordinates, sdf)
    assert deeponet(sdf, trunk_input).shape == (2, 96 * 128)
    assert fno(fno_field(sdf, coordinates, fno.architecture)).shape == (2, 1, 96, 128)


def test_chunked_gradient_equals_full_gradient():
    torch.manual_seed(1)
    model = model_for("deeponet", "baseline").double()
    sdf = torch.randn(3, 96 * 128, dtype=torch.float64)
    coordinates = torch.rand(96 * 128, 2, dtype=torch.float64)
    target = torch.rand(3, 96 * 128, dtype=torch.float64)
    mask = (torch.rand(3, 96 * 128) > 0.2).double()
    valid = mask.sum()

    # FULL BATCH
    model.zero_grad()
    prediction = model(sdf, build_case_dependent_trunk_input(coordinates, sdf))
    (((prediction - target) ** 2 * mask).sum() / valid).backward()
    full = [p.grad.clone() for p in model.parameters()]

    # CHUNKS OF 2048 POINTS, AS IN train_epoch
    model.zero_grad()
    branch = model.branch(sdf)
    for first in range(0, 96 * 128, 2048):
        last = min(first + 2048, 96 * 128)
        piece = model.forward_from_branch(
            branch, build_case_dependent_trunk_input(coordinates[first:last], sdf[:, first:last]))
        loss = ((piece - target[:, first:last]) ** 2 * mask[:, first:last]).sum() / valid
        loss.backward(retain_graph=last < 96 * 128)
    chunked = [p.grad.clone() for p in model.parameters()]

    for a, b in zip(full, chunked):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)
