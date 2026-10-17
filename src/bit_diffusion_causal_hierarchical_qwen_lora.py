"""
https://github.com/lucidrains/denoising-diffusion-pytorch

Hierarchical diffusion with a LoRA-tuned Qwen VLM encoder. The VLM is chosen by `vlm`:
  qwen2vl_7b  : was bit_diffusion_causal_hierarchical_qwen_onlylora_7b_onlyvlm.py
  qwen25vl_3b : was bit_diffusion_causal_hierarchical_qwen_onlylora_25b_darail2.py
  qwen25vl_7b : was bit_diffusion_causal_hierarchical_qwen_onlylora_7b_nturgbd.py
"""
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import nn
from tqdm import tqdm
import clip

from bit_diffusion_common import GaussianBitDiffusionBase, ModelPrediction, default, extract
from goals import goal_texts
from qwen_lora_encoder import QwenVLOnlineEncoder


# vlm -> encoder preset / checkpoint / LoRA rank / hidden size / alpha-gated goal fusion
VLM_CONFIGS = {
    'qwen2vl_7b':  dict(preset='qwen2vl',  model_id="Qwen/Qwen2-VL-7B-Instruct",   lora_rank=4, hidden_dim=3584, alpha_gate=True),
    'qwen25vl_3b': dict(preset='qwen25vl', model_id="Qwen/Qwen2.5-VL-3B-Instruct", lora_rank=8, hidden_dim=2048, alpha_gate=False),
    'qwen25vl_7b': dict(preset='qwen25vl', model_id="Qwen/Qwen2.5-VL-7B-Instruct", lora_rank=8, hidden_dim=3584, alpha_gate=False),
}


class GaussianBitDiffusion(GaussianBitDiffusionBase):
    def __init__(
        self,
        model: nn.Module,
        goalmodel:nn.Module,
        condition_x0:False,
        clip_model,
        *,
        num_classes=48,
        timesteps=1000,  
        ddim_timesteps=100,
        betas=None,
        loss_type="l2",
        objective="pred_x0",
        beta_schedule="cosine",
        vlm="qwen2vl_7b",
        goal_set="darai",
    ):
        
        super().__init__()
        self._init_diffusion(model, goalmodel, condition_x0, clip_model, num_classes, timesteps,
                             ddim_timesteps, betas, loss_type, objective, beta_schedule)
        self.goal_set = goal_set
        vlm_cfg = VLM_CONFIGS[vlm]
        self.alpha_gate = vlm_cfg["alpha_gate"]
        

        self.qwen_enc = QwenVLOnlineEncoder(
            preset=vlm_cfg["preset"],
            model_id=vlm_cfg["model_id"],
            device="cuda",
            torch_dtype=torch.bfloat16,
            cache_dir="/home/hice1/skim3513/scratch/causdiff/hf_cache",
            lora_rank=vlm_cfg["lora_rank"],
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=("q_proj","k_proj","v_proj","o_proj"),
            freeze_vision_tower=True,
            freeze_mm_projector=True,#False,
            max_side=448
        )

        # === NEW: 프로젝션 헤드 & 로짓 스케일 (게으른 초기화) ===
        self.proj_vlm = nn.Linear(vlm_cfg["hidden_dim"], 512, bias=False)
        self.proj_diff = nn.Linear(512, 512, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(3.0))  # CLIP-style; exp(3)≈20
        self.lambda_contrast = 1#0.5
        if self.alpha_gate:
            self.alpha_logit = torch.nn.Parameter(torch.zeros(512))
    
    def _ensure_proj_heads(self, vlm_latents, diff_vecs, out_dim=512):
        # vlm_latents: (B,T,Dv)
        # diff_vecs  : (B,T,Dd)
        
        Dv = vlm_latents.shape[-1]
        Dd = diff_vecs.shape[-1]
        print(Dv, Dd) #3584, 512
        if self.proj_vlm is None:
            self.proj_vlm = nn.Linear(Dv, out_dim, bias=False).to(vlm_latents.device)
        if self.proj_diff is None:
            self.proj_diff = nn.Linear(Dd, out_dim, bias=False).to(diff_vecs.device)

    def _info_nce_timewise(self, q, k, mask=None, temperature=0.07):
        """
        q, k: (B,T,D), 동일 시점끼리 positive, 배치 내 나머지는 negative
        mask: (B,T) 0/1 유효 마스크
        """
        
        B,T,D = q.shape
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k[:, :T, :], dim=-1)

        qf = q.reshape(B*T, D)      # (N,D)
        kf = k.reshape(B*T, D)      # (N,D)
        logits = qf @ kf.t() / temperature   # (N,N)
        targets = torch.arange(B*T, device=q.device)

        # if mask is not None:
        #     m = mask.reshape(B*T).bool()
        #     idx = torch.nonzero(m, as_tuple=False).squeeze(1)
            
        #     logits = logits[idx][:, idx]
            
        #     targets = torch.arange(idx.numel(), device=q.device)

        loss = F.cross_entropy(logits, targets)
        return loss

    










    # SAMPLE from q(x_t | x_o)

  
    # PREDICTION and LOSSES
    def p_losses(self,
                t,
                x_0,
                obs,
                mask_all,
                mask_past,gt_goal=None,gt_goal_one_hot=None,
                noise=None, images_paths_batch=None,):
        condition = False
        # SAMPLE x_t from q(x_t | x_o)
        
        ############## GUESS LABEL ###################
        x_start = x_0
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # sample: goal
        B, T, _ = gt_goal_one_hot.shape
        
        global_goal_classes = gt_goal_one_hot[:,-1].argmax(dim=-1)  # (B,)
        global_goal_texts = goal_texts(self.goal_set, global_goal_classes)

        # CLIP Tokenization
        goal_tokens = clip.tokenize(global_goal_texts).to(gt_goal.device)
        goal_features = self.clip.encode_text(goal_tokens)  # (B, D_clip)
        if goal_features.shape[0] != B:
            goal_features = goal_features.expand(B, -1)

        # 2. 확장 (time axis 맞추기)
        goal_start = goal_features.unsqueeze(1).expand(-1, T, -1)  # (B, T, D_clip)
        
        #goal_start = gt_goal_one_hot # (16, 1816, 10)
        goal_noise = None
        goal_noise = default(goal_noise, lambda: torch.randn_like(goal_start))
        
        goal_t = self.q_sample(x_start=goal_start, t=t, noise=goal_noise)

        # OBSERVATION CONDITIONING
        obs_cond = obs * mask_past

        # SELF-CONDITIONING
        #self_cond = torch.zeros_like(gt_goal_one_hot).to(gt_goal_one_hot.device)
        self_cond = torch.zeros((gt_goal_one_hot.shape[0], gt_goal_one_hot.shape[1], 512), device=gt_goal_one_hot.device)#384
        flag=False
        if torch.rand((1)) < 0.5 and self.condition_x0:
            flag = True
            with torch.no_grad():
                _, infer_goal = self.goalmodel(
                    x=goal_t, 
                    t=t, 
                    stage_masks=mask_all,
                    obs_cond=obs_cond, # frames
                    self_cond=self_cond,
                )
                self_cond = self.attn(infer_goal[-1], goal_features) # (B, T, D)
                #self_cond = infer_goal[-1].detach()

        # REVERSE STEP
        _, model_out_goal = self.goalmodel(
            x=goal_t,
            t=t,
            stage_masks=mask_all,
            obs_cond=obs_cond,
            self_cond=self_cond,
        )  # S x B x T x C

        S, B, T, D = model_out_goal.shape

        subgoal_seq = model_out_goal # (S x B x T x C)
        goal_repeat = subgoal_seq[-1] # (B x T x C)

        vlm_latents = self.qwen_enc(
            images=images_paths_batch,
            global_intention=global_goal_texts,
            frame_indices=None, total_len=None, recent_subs_batch=None
        )  # (B,T,Dv)
        #self._ensure_proj_heads(vlm_latents, goal_repeat)  # lazy init
        if flag:
            q_vlm = self.proj_vlm(vlm_latents)                 # grad ON
            k_diff = self.proj_diff(goal_repeat.detach())      # grad OFF
            #self.lambda_contrast = 0.01
        else:
            q_vlm = self.proj_vlm(vlm_latents.detach())                 # grad OFF
            k_diff = self.proj_diff(goal_repeat)      # grad ON

        valid = mask_all[-1].squeeze(-1).to(q_vlm.dtype)   # (B,T)
        temp = torch.exp(self.logit_scale).clamp(1/100.0, 100.0).detach()
        nce  = self._info_nce_timewise(q_vlm, k_diff, mask=valid, temperature=float(1.0/temp))


        if self.alpha_gate:
            alpha = torch.sigmoid(self.alpha_logit).view(1, 1, D).to(dtype=q_vlm.dtype, device=q_vlm.device)
            goal_repeat = alpha * k_diff + (1.0 - alpha) * q_vlm

        #goal_logits = model_out_goal.mean(dim=2, keepdim=True) # (S, B, 1, C)
        gt_goal_one_hot = gt_goal_one_hot[:, :1, :]
        # SELF-CONDITIONING
        if torch.rand((1)) < 0.5 and self.condition_x0:
            with torch.no_grad():
                self_cond = torch.zeros((model_out_goal.shape[1], model_out_goal.shape[2], x_0.shape[2]+model_out_goal.shape[3]), device=model_out_goal.device)
                self_cond, _ = self.model(
                    x=x_t, 
                    t=t, 
                    stage_masks=mask_all,
                    obs_cond=obs_cond, 
                    self_cond=self_cond
                )
                self_cond = self_cond[-1]
                self_cond = self_cond.detach()
        else:
            self_cond = torch.zeros_like(x_0).to(x_0.device)
            
        
        self_cond = torch.cat([self_cond, goal_repeat], dim=2) # (b,t,2c)
        # REVERSE STEP
        model_out, _ = self.model(
            x=x_t,
            t=t,
            stage_masks=mask_all,
            obs_cond=obs_cond,
            self_cond=self_cond,
        )  # S x B x T x C

        # LOSS
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_0 
        else:
            raise ValueError(f"unknown objective {self.objective}")
        

        # KL (q(x_t-1 | x_o, x_t) || p(x_t-1 | x_t))
        if self.loss_type == 'l2':
            target = repeat(target, 'b t c -> s b t c', s=model_out.shape[0])
            
            mask_all = torch.stack(mask_all, dim=0)
            loss = self.loss_fn(model_out, target, reduction="none")  # S x B x T x C
            loss = torch.sum(torch.mean(loss * mask_all, dim=(2, 3)))
            if gt_goal is not None:
                ### goal diffusion: goal_logits (S x B x 1 x C)
                ### action diffusion: model_out (S x B x T x C)
                loss_goal = self.semantic_consistency_loss(
                    subgoal_features=subgoal_seq,
                    global_goal=goal_start
                )
                #loss_goal = torch.sum(torch.mean(loss_goal * mask_all, dim=(2, 3)))
                loss += loss_goal
            loss = loss + self.lambda_contrast * nce


        # OUT
        return loss, rearrange(model_out, 's b t c -> s b c t')




    # ---------------------------------- INFERENCE (DDIM) --------------------------------------

    def model_predictions(self, x, pred_x_start_prev, goal_x, goal_pred_x_start_prev, t, obs, stage_masks, gt_goal=None, gt_goal_one_hot=None, index=0, video_file=None,):
        x_t = x
        B,T,C=gt_goal_one_hot.shape
        # Given x_t, reconsturct x_0
        #self_cond_goal = torch.zeros_like(gt_goal_one_hot).to(gt_goal_one_hot.device)
        self_cond = torch.zeros_like(pred_x_start_prev).to(pred_x_start_prev.device)
        #self_cond = torch.zeros((B, T, 58), device=pred_x_start_prev.device)
        if self.condition_x0:
            self_cond = pred_x_start_prev
        
        ###################################################################################
        #goal_t = gt_goal_one_hot#torch.randn((B, T, C)).to(pred_x_start_prev.device)  # e.g., (16, 1, 48) shaped random noise
        goal_t = goal_x
        _, T, _ = gt_goal_one_hot.shape
        global_goal_classes = gt_goal_one_hot[:,-1].argmax(dim=-1)  # (B,)
        global_goal_texts = goal_texts(self.goal_set, global_goal_classes)
        
        ## CLIP tokenization
        goal_tokens = clip.tokenize(global_goal_texts).to(x.device)
        goal_features = self.clip.encode_text(goal_tokens)  # (B, D_clip)
        #goal_t = goal_features.unsqueeze(1).expand(-1, T, -1)  # (B, T, D_clip)
        if goal_features.shape[0] != B:
            goal_features = goal_features.expand(B, -1)
        self_cond_goal = self.attn(goal_pred_x_start_prev, goal_features)
        
        _, infer_goal = self.goalmodel(
            x=goal_t, 
            t=t, 
            stage_masks=stage_masks,
            obs_cond=obs,
            self_cond = self_cond_goal,index=index
        )
        
        # vlm_latents = self.qwen_enc(
        #     images=video_file,
        #     global_intention=global_goal_texts[0],
        #     frame_indices=None, total_len=None, recent_subs_batch=None
        # )
        
        # ################### npy save ###########################################
        # if t[0] == 0:
        #     save_dir = "outputs/utkinects/infer_goal"
        #     os.makedirs(save_dir, exist_ok=True)
        #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     np.save(os.path.join(save_dir, f"infer_goal_t{int(t[0])}_{ts}.npy"), infer_goal.detach().to('cpu').numpy().astype('float32', copy=False))
        # ################### npy save end ###########################################

        # def _slug(s: str) -> str:
        #     import re
        #     s = s.strip()
        #     s = re.sub(r"[^\w\s-]", "", s)    # 특수문자 제거
        #     s = re.sub(r"\s+", "_", s)        # 공백 -> _
        #     return s[:60] if len(s) > 60 else s

        # if t[0] == 0:
        #     goal_tag = _slug(global_goal_texts[0])
        #     txt_path = os.path.join("outputs/utkinects/phrases", f"phrases_{goal_tag}_{ts}.txt")
        #     phrases_bt = self.qwen_enc.generate_phrases(
        #         images=video_file,           # List[List[PIL.Image]]
        #         global_intention=global_goal_texts[0], # str or List[str]
        #         do_sample=False, num_beams=2         # 재현성 우선 + 약간의 안정성
        #     )
        #     with open(txt_path, "w", encoding="utf-8") as f:
        #         f.write(f"# global_goal: {global_goal_texts[0]}\n")
        #         f.write(f"# saved_at: {ts}\n")
        #         for b in range(len(phrases_bt)):
        #             for j in range(len(phrases_bt[b])):
        #                 line = f"time: {j}\t{phrases_bt[b][j]}"
        #                 f.write(line + "\n")

        infer_goal = infer_goal[-1] # (25, 106, 512)
        
        ###################################################################################
        ###################################################################################
        #vlm_latents = self.proj_vlm(vlm_latents)
        #alpha = torch.sigmoid(self.alpha_logit).view(1, 1, 512).to(dtype=vlm_latents.dtype, device=vlm_latents.device)
        #infer_goal = alpha * infer_goal + (1.0 - alpha) * vlm_latents.expand(25, -1, -1)
        
        ###################################################################################
        ###################################################################################

        self_cond = torch.cat([self_cond, infer_goal], dim=2)
        
        # ##################################################################################
        
        # PRED
        model_output,model_feature = self.model(
            x=x_t,
            t=t,
            stage_masks=stage_masks,
            obs_cond=obs,
            self_cond=self_cond,index=index
        )
        ################### npy save ###########################################
        # if t[0] == 0:
        #     save_dir = "outputs/utkinects/infer_action"
        #     os.makedirs(save_dir, exist_ok=True)
        #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     #np.save(os.path.join(save_dir, f"infer_action_feature_t{int(t[0])}_{ts}.npy"), model_feature.detach().to('cpu').numpy().astype('float32', copy=False))
        #     np.save(os.path.join(save_dir, f"infer_action_t{int(t[0])}_{ts}.npy"), model_output.detach().to('cpu').numpy().astype('float32', copy=False))
        ################### npy save end ###########################################
        model_output = model_output[-1]
        
        if self.objective == "pred_noise":
            pred_noise = model_output 
            pred_x_start = self.predict_start_from_noise(x, t, pred_noise) * stage_masks[-1]
            
        elif self.objective == "pred_x0":
            pred_x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, pred_x_start) * stage_masks[-1]

        pred_goal_start = infer_goal
        pred_goal_noise = self.predict_noise_from_start(goal_t, t, pred_goal_start) * stage_masks[-1]
            
        return ModelPrediction(pred_noise, pred_x_start, pred_goal_noise, pred_goal_start)



    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        pred_x_start_prev, goal_x, goal_pred_x_start_prev,
        t,
        t_prev,
        batch,
        if_prev=False,
        index=0
    ):
        gt_goal_one_hot = batch['goal_one_hot']
        # MODEL PRED
        preds = self.model_predictions(x=x,
                                       pred_x_start_prev=pred_x_start_prev,
                                       goal_x=goal_x,
                                       goal_pred_x_start_prev=goal_pred_x_start_prev,
                                       t=t,
                                       obs=batch['obs'] * batch['mask_past'],
                                       stage_masks=batch['mask_all'], gt_goal=batch['goal'],gt_goal_one_hot=gt_goal_one_hot, index=index, video_file=batch["video_file"])
        pred_x_start = preds.pred_x_start
        pred_noise = preds.pred_noise

        pred_goal_start = preds.pred_goal_start
        pred_goal_noise = preds.pred_goal_noise

        # PRED X_0
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        if if_prev:
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, x.shape)
        else:
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)
        sigma = (
                self.eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Compute mean and var
        noise = torch.randn_like(x) 
        mean_pred = (
                pred_x_start * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_noise
        )

        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))


        # PRED goal
        alpha_bar_goal = extract(self.alphas_cumprod, t, gt_goal_one_hot.shape)
        if if_prev:
            alpha_bar_prev_goal = extract(self.alphas_cumprod_prev, t_prev, gt_goal_one_hot.shape)
        else:
            alpha_bar_prev_goal = extract(self.alphas_cumprod, t_prev, gt_goal_one_hot.shape)
        sigma_goal = (
                self.eta
                * torch.sqrt((1 - alpha_bar_prev_goal) / (1 - alpha_bar_goal))
                * torch.sqrt(1 - alpha_bar_goal / alpha_bar_prev_goal)
        )

        # Compute mean and var
        #noise_goal = torch.randn_like(gt_goal_one_hot) 
        noise_goal = torch.randn((gt_goal_one_hot.shape[0], gt_goal_one_hot.shape[1], 512), device=gt_goal_one_hot.device)
        mean_pred_goal = (
                pred_goal_start * torch.sqrt(alpha_bar_prev_goal)
                + torch.sqrt(1 - alpha_bar_prev_goal - sigma_goal ** 2) * pred_goal_noise
        )

        nonzero_mask_goal = (1 - (t == 0).float()).reshape(gt_goal_one_hot.shape[0], *((1,) * (len(gt_goal_one_hot.shape) - 1)))
        return mean_pred + nonzero_mask * sigma * noise, pred_x_start, mean_pred_goal+nonzero_mask_goal*sigma_goal*noise_goal, pred_goal_start

   

    @torch.no_grad()
    def p_sample_loop_with_input(
        self,
        batch,
        init_rand=None,
        n_diffusion_steps=-1,index=0
    ):
        
        # INPUT
        device = self.betas.device
        x_0_pred = torch.zeros_like(batch["x_0"]).to(batch["x_0"].device)  # only used for shape
        #goal_x_0_pred = torch.zeros_like(batch["goal_one_hot"]).to(batch["goal_one_hot"].device)  # only used for shape
        goal_x_0_pred = torch.zeros((batch["goal_one_hot"].shape[0], batch["goal_one_hot"].shape[1], 512), device=batch["goal_one_hot"].device)
    
        # INIT PREDICTION (normal distr noise)
        pred = torch.randn_like(x_0_pred, device=device) if init_rand is None else init_rand  # BS x T x C
        init_noise = pred.clone()
        pred = pred.contiguous()
        goal_pred = torch.randn_like(goal_x_0_pred, device=device) if init_rand is None else init_rand  # BS x T x C
        goal_pred = goal_pred.contiguous()

        # SAMPLE
        assert n_diffusion_steps == len(self.ddim_timestep_seq) # 50
        
        # Resample (DDIM)
        for t in tqdm(
            reversed(range(0, n_diffusion_steps)),
            desc="Resampled sampling loop time step",
            total=n_diffusion_steps,
            position=0,
            leave=True
        ):

            batched_times = torch.full((pred.shape[0],), self.ddim_timestep_seq[t], device=pred.device, dtype=torch.long)
            if t == 0:
                batched_times_prev = torch.full((pred.shape[0],), 0, device=device, dtype=torch.long)
                pred, x_0_pred, goal_pred, goal_x_0_pred = self.p_sample_ddim(
                    x=pred, 
                    pred_x_start_prev=x_0_pred,
                    goal_x=goal_pred,
                    goal_pred_x_start_prev=goal_x_0_pred,
                    t=batched_times, 
                    t_prev=batched_times_prev, 
                    batch=batch,
                    if_prev=True,index=index
                )
            else:
                batched_times_prev = torch.full((pred.shape[0],), self.ddim_timestep_seq[t-1], device=device, dtype=torch.long)
                pred, x_0_pred, goal_pred, goal_x_0_pred = self.p_sample_ddim(
                    x=pred,
                    pred_x_start_prev=x_0_pred, 
                    goal_x=goal_pred,
                    goal_pred_x_start_prev=goal_x_0_pred,
                    t=batched_times,
                    t_prev=batched_times_prev,
                    batch=batch,index=index
                )
        return pred, init_noise



    ''' Actual inference step '''
    def predict(
        self,
        x_0,
        obs,
        mask_past,
        masks_stages,
        *,
        n_samples=2,
        return_noise=False,
        n_diffusion_steps=-1,
        goal=None,gt_goal_one_hot=None,index=0, video_name=None,
    ):
        
        # Initialize observation
        
        obs = repeat(obs, "b t c -> (s b) t c", s=n_samples)
        x_0 = repeat(x_0, "b t c -> (s b) t c ", s=n_samples)
        mask_past = repeat(mask_past, "b t 1 -> (s b) t c", s=n_samples, c=obs.shape[-1])
        masks_stages = [repeat(mask.to(torch.bool), "b t c -> (s b) t c", s=n_samples) for mask in masks_stages]
        #goal = repeat(goal, "b c -> (s b) c", s=n_samples)
        gt_goal_one_hot = repeat(gt_goal_one_hot, "b t c -> (s b) t c", s=n_samples)
        
        # Sample from the diffusion model
        
        x_out, init_noise = self.p_sample_loop_with_input(
            batch={
                "x_0": x_0,  # only used for shape
                "obs": obs,
                "mask_past": mask_past.to(torch.bool),
                "mask_all": masks_stages,
                "goal": None,#goal,
                "goal_one_hot":gt_goal_one_hot,
                "video_file": video_name,
            },
            init_rand=None,
            n_diffusion_steps=n_diffusion_steps,index=index
        )

        #save_matrix_npy(np.asarray(x_out.cpu()), index=index)
          
        # Return
        init_noise = init_noise[0]
        if return_noise:
            assert n_samples == 1
            return x_out, init_noise
        return rearrange(x_out, "(s b) t c -> s b c t", s=n_samples)


