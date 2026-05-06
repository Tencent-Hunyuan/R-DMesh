import torch
import torch.nn.functional as F
from torchdiffeq import odeint
from tqdm import tqdm

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.flatten(1).mean(1)

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(torch.pi / 2 * t) + 1))

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

def rf_training_losses_misalign(model, x_start, model_kwargs=None, noise=None, x0_channels=32, x1_channels=32, xt_channels=32, predict="flow"):
    """
    Compute training losses for a single timestep.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if noise is None:
        noise = torch.randn_like(x_start)
    times = torch.rand(x_start.shape[0], device=x_start.device)
    padded_times = append_dims(times, x_start.ndim - 1)
    t = cosmap(padded_times)
    x_t = t * x_start + (1. - t) * noise
    x_t = torch.cat([x_start[..., :x0_channels], x_t[..., x0_channels:]], dim=-1)
    flow = x_start - noise
    terms = {}
    model_output = model(x_t, times, **model_kwargs)
    if predict == 'flow':
        target = flow
    elif predict == 'noise':
        target = noise
    else:
        raise ValueError(f'unknown objective {predict}')
    mse_x1 = (target[..., x0_channels:x0_channels+x1_channels] - model_output[..., x0_channels:x0_channels+x1_channels]) ** 2
    mse_xt = (target[..., -xt_channels:] - model_output[..., -xt_channels:]) ** 2
    terms["mse_x1"] = mean_flat(mse_x1)
    terms["mse_xt"] = mean_flat(mse_xt)
    terms["mse"] = terms["mse_x1"] + terms["mse_xt"]
    terms["loss"] = terms["mse"]
    return terms

@torch.no_grad()
def rf_sample_vc_misalign(
    model,
    shape,
    steps=64,
    model_kwargs=None,
    device=None,
    guidance_scale=3.0,
    predict='flow',
    x0=None,
    x1=None
):   
    x0_channels = x0.shape[-1]
    odeint_kwargs = dict(
        atol = 1e-5,
        rtol = 1e-5,
        method = 'midpoint'
    )
    model_kwargs['vid_embed'] = torch.cat([model_kwargs['vid_embed'], torch.zeros(model_kwargs['vid_embed'].shape, device=device)], dim=0)
    x0_double = torch.cat([x0] * 2)
    if x1 is not None:
        x1_channels = x1.shape[-1]
        x1_double = torch.cat([x1] * 2)
    def ode_fn(t, x):
        x = torch.cat([x] * 2)
        x = torch.cat([x0_double, x[..., x0_channels:]], dim=-1)
        if x1 is not None:
            x[..., x0_channels:x0_channels+x1_channels] = x1_double
        flow = model(x, t.unsqueeze(0).repeat(x.shape[0]), **model_kwargs) 
        # cfg
        cond_flow, uncond_flow = torch.split(flow, len(flow) // 2, dim=0)
        flow = uncond_flow + guidance_scale * (cond_flow - uncond_flow)
        return flow
    noise = torch.randn(*shape, device=device) 
    times = torch.linspace(0., 1., steps, device=device)
    # ode
    trajectory = odeint(ode_fn, noise, times, **odeint_kwargs)
    sampled_data = trajectory[-1]
    sampled_data = torch.cat([x0, sampled_data[..., x0_channels:]], dim=-1)
    if x1 is not None:
        sampled_data[..., x0_channels:x0_channels+x1_channels] = x1
    return sampled_data

