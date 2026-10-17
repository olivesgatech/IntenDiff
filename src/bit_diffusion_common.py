"""
Parts shared by every bit_diffusion_causal_hierarchical_qwen* model.
https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from causal_attention import CausalAttention


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "pred_goal_noise", "pred_goal_start"])


class DiffusionModel(nn.Module):
    """
    Template model for the diffusion process
    """

    def __init__(
        self,
    ):
        super().__init__()


    def forward(self, X_0, X_t, batch):
        raise NotImplementedError("Nope")


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # B x 1 x 1


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianBitDiffusionBase(nn.Module):
    """Noise schedule, q-sampling and losses common to all hierarchical diffusion variants."""

    def _init_diffusion(
        self,
        model,
        goalmodel,
        condition_x0,
        clip_model,
        num_classes,
        timesteps,
        ddim_timesteps,
        betas,
        loss_type,
        objective,
        beta_schedule,
    ):
        print(f'Num classes : {num_classes}')
        print(f'Loss type : {loss_type}')
        print(f'Objective: {objective}')
        print(f'Beta schedule : {beta_schedule}')

        self.model = model
        self.num_classes = num_classes
        self.condition_x0 = condition_x0
        self.goalmodel = goalmodel

        # RECONSTRUCTION OBJ
        self.objective = objective
        self.loss_type = loss_type

        assert objective in {
            "pred_noise",
            "pred_x0",
        }, "objective must be either pred_noise (predict noise) or pred_x0(predict image start)"  # noqa E501


        # VARIANCE
        if betas is None:
            if beta_schedule == "linear":
                betas = linear_beta_schedule(timesteps)
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(timesteps)
            else:
                raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


        # SAMPLING
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # PARAMS
        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # FORWARD DIFFUSION q(x_t | x_{t-1})
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))


        # POSTERIOR q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)),)
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),)
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),)

        # DDIM
        self.eta = 0.
        c = timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, timesteps, c)))

        self.ddim_timesteps = ddim_timesteps
        self.ddim_timestep_seq = ddim_timestep_seq

        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False

        self.attn = CausalAttention(d_model=512, n_heads=8, num_goal_tokens=4, share_qkv=True, causal_mask=False)

    def semantic_consistency_loss(self, subgoal_features, global_goal):
        """
        subgoal_features: (S,B,T,D) - semantic features from DiffSingleStageModel
        global_goal: (B,C) - one-hot encoded global goal
        """
        S, B, T, D = subgoal_features.shape

        global_goal_features = global_goal.unsqueeze(0).expand(S,-1,-1,-1)

        # Now both tensors are (S, B, T, D)
        similarity = F.cosine_similarity(
            subgoal_features,
            global_goal_features,
            dim=-1  # compute similarity along feature dimension
        )  # (S, B, T)

        loss = 1 - similarity.mean()

        return loss

    @property
    def loss_fn(self):
        if self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
          - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )


    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)



    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
              extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )



    # SAMPLE from q(x_t | x_o)
    def q_sample(self, x_start, t, noise=None):
        """
        :param x_start: {B x T x C}
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return ( extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def forward(self, batch, *args, **kwargs):
        gt_goal = batch['goal']
        gt_goal_one_hot = batch['goal_one_hot']
        x_0 = batch['x_0']  # class labels, (16, 2996, 48) # 48 is the class number
        obs = batch['obs']  # features, (16, 2996, 2048)

        masks_stages = batch['masks_stages'] # 5-length (16, 2996, 1) each
        masks_stages = [mask.to(torch.bool) for mask in masks_stages] # 5-length (16, 2996, 1) each

        mask_past = batch['mask_past'] # (16, 2996, 1)
        mask_past = mask_past.to(torch.bool)
        mask_past = repeat(mask_past, 'b t 1 -> b t c', c=obs.shape[-1]) # (16, 2996, 2048)

        video_name = batch['video_name']

        # get random diff timestep
        t = torch.randint(0, self.num_timesteps, (obs.size(0),), device=obs.device).long() # (16)
        return self.p_losses(
            t = t,
            x_0 = x_0,
            obs = obs,
            mask_past = mask_past,
            mask_all = masks_stages,
            gt_goal=gt_goal, gt_goal_one_hot=gt_goal_one_hot, images_paths_batch=video_name,
            *args, **kwargs
        )
