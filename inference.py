import diffusers
import matplotlib.pyplot as plt
import numpy as np
import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from GNN import RGAT
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from PIL import Image
from dataset import generate_hg
import os
import random
import math
import os
import json
import argparse




def load_models(device):
    # stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable_diffusion_version = "/root/autodl-tmp/models/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
    # stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    # cache_dir="/root/autodl-tmp/models/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
    stable = StableDiffusionPipeline.from_pretrained(stable_diffusion_version).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_version, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(stable_diffusion_version, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(stable_diffusion_version, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(stable_diffusion_version, subfolder="unet").to(device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_version,
    subfolder="text_encoder").to(device)
    return stable, tokenizer, noise_scheduler, vae, unet, text_encoder, image_processor

def make_image_grid(imgs, rows, cols):
    # borrowed from https://github.com/ziqihuangg/ReVersion/blob/master/inference.py#L57
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def generate_fn(prompt, A, R, B, guidance_scale, adjustment_scale, ddim_steps):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    stable, tokenizer, noise_scheduler, vae, unet, text_encoder, image_processor = load_models(device)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.requires_grad_(False)
    gnn_model = RGAT(1024, 512, 1024, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(device)
    gnn_model.requires_grad_(False)
    embeddings = text_encoder.get_input_embeddings()

    h_graph_ARB, ARB_sentence_embedding, node_features_ARB, ARB_eot_pos = generate_hg(A, B, R, prompt, tokenizer, embeddings, text_encoder, device)
    gnn_model.eval()
    text_encoder.eval()
    save_path = "./demo_models"
    gnn_model.load_state_dict(torch.load(os.path.join(save_path, f"gat_{R}"), map_location=device)) 
    # Training finished, start generating
    seed = random.randint(1, 1000000000)

        
    guidance_scale=guidance_scale
    w=adjustment_scale

    with torch.no_grad():
        num_images_per_prompt = 4
        text_encoder.text_model.requires_grad_(False)
        img_ori = stable(prompt, guidance_scale=guidance_scale, generator=torch.Generator('cuda').manual_seed(seed), num_inference_steps=ddim_steps, num_images_per_prompt=num_images_per_prompt).images

        delta_norm = gnn_model(h_graph_ARB, node_features_ARB)['eot'][1]
        ARB_sentence_embedding[0, ARB_eot_pos] += delta_norm * w

        prompt_embeds = torch.concat([ARB_sentence_embedding])

        stable.text_encoder = text_encoder

        img_adjusted = stable(None, guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, generator=torch.Generator('cuda').manual_seed(seed), num_inference_steps=ddim_steps, num_images_per_prompt=num_images_per_prompt).images

        image_grid_ori = make_image_grid(img_ori, rows=2, cols=2)
        image_grid_adjust = make_image_grid(img_adjusted, rows=2, cols=2)
        return image_grid_ori, image_grid_adjust




            
            
            
            


