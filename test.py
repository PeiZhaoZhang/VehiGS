import torch
from diffusers import AutoencoderKL
import os

# 1. 强制禁用 cuDNN（这是解决 NOT_INITIALIZED 的唯一大招）
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False

def test_vae_standalone():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_model_name = "stabilityai/sd-vae-ft-mse" 
    
    try:
        # 保持 Tiling 和 Half，这能节省显存
        vae = AutoencoderKL.from_pretrained(vae_model_name).to(device).half()
        vae.enable_tiling() 
        vae.eval()
        print("cuDNN Disabled. VAE Tiling Enabled.")
    except Exception as e:
        print(f"Failed: {e}")
        return

    H, W = 1432, 1912
    dummy_input = torch.randn(1, 3, H, W).to(device).half()

    print("Starting VAE Encoding (Native CUDA Mode)...")
    try:
        with torch.no_grad():
            # 即使不用 cuDNN，也要保持精度一致
            posterior = vae.encode(dummy_input)
            latents = posterior.latent_dist.sample()
            
        print(f"Success! Latent shape: {latents.shape}")
        
    except RuntimeError as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_vae_standalone()