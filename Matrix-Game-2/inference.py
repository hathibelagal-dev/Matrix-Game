import torch
import numpy as np

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange
from pipeline import CausalInferencePipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import *
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file

class InteractiveGameInference:
    def __init__(self):
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load("configs/inference_yaml/inference_universal.yaml")

    def _init_models(self):
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load("/content/Wan2.1_VAE.pth", map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        pipeline = CausalInferencePipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)

        print("Loading Pretrained Model...")
        state_dict = load_file("/content/base_distilled_model/base_distill.safetensors")
        pipeline.generator.load_state_dict(state_dict)
        print("Loaded pretrained model.")

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        print("Loading VAE...")
        vae = get_wanx_vae_wrapper("/content/", torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)
        print("VAE ready.")

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image
    
    def generate_videos(self):
        mode = "universal"
        num_output_frames = 84
        image = load_image("/content/input.png")
        image = self._resizecrop(image, 352, 640)
        print("Resized input image.")
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        # Encode the input image as the first latent
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        print("Encoding input image...")
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)         
        visual_context = self.vae.clip.encode_video(image)
        print("Encoded.")
        sampled_noise = torch.randn(
            [1, 16,num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype
        )
        num_frames = (num_output_frames - 1) * 4 + 1
        
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }
        
        cond_data = Bench_actions_universal(num_frames)
        conditional_dict['mouse_cond'] = None
        conditional_dict['keyboard_cond'] = cond_data['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        
        print("Now running the inference pipeline...")
        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                mode=mode,
                profile=False
            )

        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)
        process_video(video.astype(np.uint8), '/content/gamegen.mp4')
        print("Done")

def main():
    """Main entry point for video generation."""
    pipeline = InteractiveGameInference()
    pipeline.generate_videos()

if __name__ == "__main__":
    main()