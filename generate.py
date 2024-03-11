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
from dataset import HeteGraphDataset
import os
import random
import math
import os
import json
import argparse



def load_models(device):
    # stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
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

def generate(args):
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    stable, tokenizer, noise_scheduler, vae, unet, text_encoder, image_processor = load_models(device)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.requires_grad_(False)
    gnn_model = RGAT(1024, args.hidden_size, 1024, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(device)

    gnn_model.requires_grad_(False)
    embeddings = text_encoder.get_input_embeddings()

    
    folder = args.data_folder
    config_file_path = os.path.join(folder, "index.json")
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    A = config['A']
    R = config['R']
    B = config['B']
    ARB = config['ARB']
    ARB_dir = config['ARB_dir']
    BRA_dir = config['BRA_dir']
    BRA = config['BRA']
    
    hete_dataset = HeteGraphDataset(folder, A, B, R, ARB, BRA, ARB_dir, BRA_dir, image_processor, tokenizer)
    gnn_model.eval()
    text_encoder.eval()
    
    A_img, B_img, ARB_img, BRA_img, ARB_sentence_embedding, ARB_eot_pos, BRA_sentence_embedding, BRA_eot_pos, \
    A_eot_embedding, A_sentence_embedding, A_eot_pos, B_eot_embedding, B_sentence_embedding, B_eot_pos, \
    h_graph_ARB, node_features_ARB, h_graph_BRA, node_features_BRA\
    = hete_dataset.sample(A, R, B, ARB, BRA, text_encoder, embeddings, device)
    

    # Training finished, start generating
    s = 0
    save_path = os.path.join(args.save_folder, R)
    relation_base_dir = save_path
    prompt_pair = [ARB, BRA]
    gnn_model.load_state_dict(torch.load(os.path.join(save_path, f"gat_{R}"), map_location=torch.device('cuda')))
    
    relation0_dir = os.path.join(relation_base_dir, prompt_pair[0])
    relation1_dir = os.path.join(relation_base_dir, prompt_pair[1])
    if not os.path.exists(relation0_dir):
        os.makedirs(relation0_dir)
    if not os.path.exists(relation1_dir):
        os.makedirs(relation1_dir)
        
    guidance_scale=7.5
    seeds = [random.randint(1, 1000000000) for _ in range(1000)]

    with torch.no_grad():
        num_images_per_prompt = 4
        for w in [0, 0.2, 0.4, 0.6, 0.8]:
            s = 0
            relation0_dir_w = os.path.join(relation0_dir, f"w={w}")
            if not os.path.exists(relation0_dir_w):
                os.makedirs(relation0_dir_w)
                
            relation1_dir_w = os.path.join(relation1_dir, f"w={w}")
            if not os.path.exists(relation1_dir_w):
                os.makedirs(relation1_dir_w)

            A_img, B_img, ARB_img, BRA_img, ARB_sentence_embedding, ARB_eot_pos, BRA_sentence_embedding, BRA_eot_pos, \
                A_eot_embedding, A_sentence_embedding, A_eot_pos, B_eot_embedding, B_sentence_embedding, B_eot_pos, \
                h_graph_ARB, node_features_ARB, h_graph_BRA, node_features_BRA\
                = hete_dataset.sample(A, R, B, ARB, BRA, text_encoder, embeddings, device)
            text_encoder.text_model.requires_grad_(False)
            delta_norm = gnn_model(h_graph_ARB, node_features_ARB)['eot'][1]
            delta_inver = gnn_model(h_graph_BRA, node_features_BRA)['eot'][1]
            ARB_sentence_embedding[0, ARB_eot_pos] += delta_norm * w
            BRA_sentence_embedding[0, BRA_eot_pos] += delta_inver * w

            for i in range(0, args.num_sample, num_images_per_prompt):

                prompt_embeds = torch.concat([ARB_sentence_embedding, BRA_sentence_embedding])

                stable.text_encoder = text_encoder

                img_ori = stable(None, guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, generator=torch.Generator('cuda').manual_seed(seeds[s]), num_inference_steps=30, num_images_per_prompt=num_images_per_prompt).images
                s += 1
                
                for j in range(num_images_per_prompt):
                    img_ori[j].save(os.path.join(relation0_dir_w, f"{i+j}.png"))

                for j in range(num_images_per_prompt):
                    img_ori[num_images_per_prompt+j].save(os.path.join(relation1_dir_w, f"{i+j}.png"))

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Cuda id",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="./RRdataset-v1/inside/",
        help="Path to your dataset",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="./RRdataset-v1/generation_result",
        help="Path to your saved checkpoint",
    )
    
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="The hidden size of graph neural network",
    )
    
    parser.add_argument(
        "--num_sample",
        type=int,
        default=4,
        help="The number of images to be generated",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
     args = parse_args()
     generate(args)


            
            
            
            



