import types
from typing import List, Optional
import torch
from einops import rearrange
from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.vae import _video_vae
from wan.modules.causal_model import CausalWanModel

class WanDiffusionWrapper(torch.nn.Module): 
    def __init__(
            self,
            model_config="",
            timestep_shift=5.0,
            is_causal=True,
    ):
        super().__init__()
        print(model_config)
        self.model = CausalWanModel.from_config(model_config)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 15 * 880 # 32760  # [1, 15, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None, kv_cache_mouse: Optional[List[dict]] = None, kv_cache_keyboard: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
    
        assert noisy_image_or_video.shape[1] == 16
        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        logits = None

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.to(self.model.dtype),#.permute(0, 2, 1, 3, 4),
                t=input_timestep, **conditional_dict,
                # seq_len=self.seq_len,
                kv_cache=kv_cache,
                kv_cache_mouse=kv_cache_mouse, kv_cache_keyboard=kv_cache_keyboard,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            )#.permute(0, 2, 1, 3, 4)
            
        else:
            flow_pred = self.model(
                noisy_image_or_video.to(self.model.dtype),#.permute(0, 2, 1, 3, 4),
                t=input_timestep, **conditional_dict)
            #.permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=rearrange(flow_pred, 'b c f h w -> (b f) c h w'),#.flatten(0, 1),
            xt=rearrange(noisy_image_or_video, 'b c f h w -> (b f) c h w'),#.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        )# .unflatten(0, flow_pred.shape[:2])
        pred_x0 = rearrange(pred_x0, '(b f) c h w -> b c f h w', b=flow_pred.shape[0])
        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

