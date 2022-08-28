import torch
import torch.nn.functional as F


def point_sample(input, point_coords, **kwargs):
    """
    From Detectron2, point_features.py#19

    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    """
    Follows 3.1. Point Selection for Inference and Training

    In Train:, `The sampling strategy selects N points on a feature map to train on.`

    In Inference, `then selects the N most uncertain points`

    Args:
        mask(Tensor): [B, C, H, W]
        N(int): `During training we sample as many points as there are on a stride 16 feature map of the input`
        k(int): Over generation multiplier
        beta(float): ratio of importance points
        training(bool): flag

    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points, 2]
    """
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    device = mask.device
    B, _, H, W = mask.shape
    mask, _ = mask.sort(1, descending=True)

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step
        # points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        points[:, :, 1] = H_step / 2.0 + (torch.div(idx,W,rounding_mode='floor')).to(torch.float) * H_step
        return idx, points

    # Official Comment : point_features.py#92
    # It is crucial to calculate uncertanty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to worse results. To illustrate the difference: a sampled point between two coarse predictions
    # with -1 and 1 logits has 0 logit prediction and therefore 0 uncertainty value, however, if one
    # calculates uncertainties for the coarse predictions first (-1 and -1) and sampe it for the
    # center point, they will get -1 unceratinty.

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertainty_map.topk(int(beta * N), -1)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)
