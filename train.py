import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from GNN import RGAT
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

def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))
    
def get_paired_latent(vae, img1, img2, device, bsz=1):
    latents0 = vae.encode(img1.to(device)).latent_dist.sample().detach()
    latents0 = latents0 * vae.config.scaling_factor
    latents0 = torch.cat([latents0] * bsz)
    
    latents1 = vae.encode(img2.to(device)).latent_dist.sample().detach()
    latents1 = latents1 * vae.config.scaling_factor
    latents1 = torch.cat([latents1] * bsz)
    
    latents_norm = torch.cat([latents0, latents1]).to(device)
    latents_inver = torch.cat([latents1, latents0]).to(device)
    
    
    
    
    return latents_norm, latents_inver

def get_loss_of_one_pair(vae, gnn_model, noise_scheduler, unet, img1, img2, 
                         sentence1_embedding, sentence1_eot_pos, sentence1_type,
                         sentence2_embedding, sentence2_eot_pos, sentence2_type,
                         h_graph1, node_features1,
                         h_graph2, node_features2,
                         device, bsz=1, importance_sampling=True, noise_weight=0.05):
    
    # The importance sampling is applied, borrowed from https://github.com/ziqihuangg/ReVersion
    if importance_sampling:
            list_of_candidates = [
                x for x in range(noise_scheduler.config.num_train_timesteps)
            ]
            prob_dist = [
                importance_sampling_fn(x,
                                       noise_scheduler.config.num_train_timesteps,
                                       0.5)
                for x in list_of_candidates
            ]
            prob_sum = 0
            for i in prob_dist:
                prob_sum += i
            prob_dist = [x / prob_sum for x in prob_dist]
    
    timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps, (bsz, ),
                    device=device)
    timesteps = timesteps.long()
    timesteps = torch.cat([timesteps, timesteps]).to(device)

    if importance_sampling:
            timesteps = np.random.choice(
                list_of_candidates,
                size=bsz,
                replace=True,
                p=prob_dist)
            timesteps = torch.tensor(timesteps).cuda()
        
    latents_norm, latents_inver = get_paired_latent(vae, img1, img2, device, bsz)
    noise = torch.randn_like(latents_norm).to(device)
    
    noisy_latents_norm = noise_scheduler.add_noise(
                        latents_norm, noise, timesteps)
    noisy_latents_inver = noise_scheduler.add_noise(
                        latents_inver, noise, timesteps)
    
    
    sentence1_embedding = torch.cat([sentence1_embedding] * bsz)
    sentence2_embedding = torch.cat([sentence2_embedding] * bsz)
    
    if sentence1_type == "r":
        delta1 = gnn_model(h_graph1, node_features1)['eot'][1]
    else:
        
        delta1 = gnn_model(h_graph1, node_features1)['eot'][0]
        noise_eot = torch.randn_like(delta1)
        delta1 += noise_weight * noise_eot
        # for object, the eot is the first one 
    if sentence2_type == "r":
        delta2 = gnn_model(h_graph2, node_features2)['eot'][1]
    else:
        delta2 = gnn_model(h_graph2, node_features2)['eot'][0]
        noise_eot = torch.randn_like(delta2)
        delta2 += noise_weight * noise_eot
        # for object, the eot is the first one 
        
    sentence1_embedding[:, sentence1_eot_pos] += delta1
    sentence2_embedding[:, sentence2_eot_pos] += delta2
    
    
    emb_pair = torch.cat([sentence1_embedding, sentence2_embedding])
    
    model_pred = unet(noisy_latents_norm, timesteps, emb_pair).sample
    model_pred_inv = unet(noisy_latents_inver, timesteps, emb_pair).sample
    
    denoise_loss = F.mse_loss(
                        model_pred.float(), noise.float(), reduction="mean")
    neg_loss = F.mse_loss(
                        model_pred_inv.float(), noise.float(), reduction="mean")
    
    cosine_sim_eot = F.cosine_similarity(sentence1_embedding[0][sentence1_eot_pos], sentence2_embedding[0][sentence2_eot_pos], dim=0)
    
    loss = 10*denoise_loss - 2*neg_loss
    
    return loss, denoise_loss, neg_loss, cosine_sim_eot.item()

def train(args):
# def train(device=1, num_sample=50, save_folder="./generation_result", data_folder="./RRdataset-v1/inside"):
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    stable, tokenizer, noise_scheduler, vae, unet, text_encoder, image_processor = load_models(device)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.requires_grad_(False)
    gnn_model = RGAT(1024, args.hidden_size, 1024, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(device)

    gnn_model.requires_grad_(True)
    embeddings = text_encoder.get_input_embeddings()

    optimizer = torch.optim.AdamW(
        gnn_model.parameters(
        ), 
        lr=args.lr, weight_decay=0.01
    )
    
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
    
    ARB_BRA_sim = []
    ARB_A_sim = []
    ARB_B_sim = []
    BRA_A_sim = []
    BRA_B_sim = []
    
    for epoch in range(0, args.epochs):
        
        print(f"epoch {epoch}:")
        
        gnn_model.train()
        text_encoder.eval()
        
        A_img, B_img, ARB_img, BRA_img, ARB_sentence_embedding, ARB_eot_pos, BRA_sentence_embedding, BRA_eot_pos, \
        A_eot_embedding, A_sentence_embedding, A_eot_pos, B_eot_embedding, B_sentence_embedding, B_eot_pos, \
        h_graph_ARB, node_features_ARB, h_graph_BRA, node_features_BRA\
        = hete_dataset.sample(A, R, B, ARB, BRA, text_encoder, embeddings, device)
        loss, denoise_loss, neg_loss, cosine_sim_eot = get_loss_of_one_pair(vae, gnn_model, noise_scheduler, unet,ARB_img, BRA_img, 
                            ARB_sentence_embedding, ARB_eot_pos, "r",
                            BRA_sentence_embedding, BRA_eot_pos, "r",
                            h_graph_ARB, node_features_ARB,
                            h_graph_BRA, node_features_BRA,
                            device, bsz=1)
        print(f"ARB BRA denoise_loss: {denoise_loss:.3f}, neg_loss: {neg_loss:.3f}")
        print(f"cosine_sim_eot, {cosine_sim_eot:.3f}")
        ARB_BRA_sim.append(cosine_sim_eot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # ARB A
        loss, denoise_loss, neg_loss, cosine_sim_eot = get_loss_of_one_pair(vae, gnn_model, noise_scheduler, unet,ARB_img, A_img, 
                            ARB_sentence_embedding, ARB_eot_pos, "r",
                            A_sentence_embedding, A_eot_pos, "o",
                            h_graph_ARB, node_features_ARB,
                            h_graph_ARB, node_features_ARB,
                            device, bsz=1)
        print(f"ARB A denoise_loss: {denoise_loss:.3f}, neg_loss: {neg_loss:.3f}")
        print(f"cosine_sim_eot, {cosine_sim_eot:.3f}")
        ARB_A_sim.append(cosine_sim_eot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # ARB B
        loss, denoise_loss, neg_loss, cosine_sim_eot = get_loss_of_one_pair(vae, gnn_model, noise_scheduler, unet,ARB_img, B_img, 
                            ARB_sentence_embedding, ARB_eot_pos, "r",
                            B_sentence_embedding, B_eot_pos, "o",
                            h_graph_ARB, node_features_ARB,
                            h_graph_BRA, node_features_BRA,
                            device, bsz=1)
        print(f"ARB B denoise_loss: {denoise_loss:.3f}, neg_loss: {neg_loss:.3f}")
        print(f"cosine_sim_eot, {cosine_sim_eot:.3f}")
        ARB_B_sim.append(cosine_sim_eot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # BRA B
        loss, denoise_loss, neg_loss, cosine_sim_eot = get_loss_of_one_pair(vae, gnn_model, noise_scheduler, unet,BRA_img, B_img, 
                            BRA_sentence_embedding, BRA_eot_pos, "r",
                            B_sentence_embedding, B_eot_pos, "o",
                            h_graph_BRA, node_features_BRA,
                            h_graph_BRA, node_features_BRA,
                            device, bsz=1)
        print(f"BRA B denoise_loss: {denoise_loss:.3f}, neg_loss: {neg_loss:.3f}")
        print(f"cosine_sim_eot, {cosine_sim_eot:.3f}")
        BRA_B_sim.append(cosine_sim_eot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # BRA A
        loss, denoise_loss, neg_loss, cosine_sim_eot = get_loss_of_one_pair(vae, gnn_model, noise_scheduler, unet,BRA_img, A_img, 
                            BRA_sentence_embedding, BRA_eot_pos, "r",
                            A_sentence_embedding, A_eot_pos, "o",
                            h_graph_BRA, node_features_BRA,
                            h_graph_ARB, node_features_ARB,
                            device, bsz=1)
        print(f"BRA A denoise_loss: {denoise_loss:.3f}, neg_loss: {neg_loss:.3f}")
        print(f"cosine_sim_eot, {cosine_sim_eot:.3f}")
        BRA_A_sim.append(cosine_sim_eot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    

    iterations = list(range(1, len(ARB_BRA_sim) + 1))
    
    plt.plot(iterations, ARB_BRA_sim, label='ARB_BRA_sim')


    plt.xlabel('epoch')
    plt.ylabel('EOT similarity')
    plt.legend()


    save_path = os.path.join(args.save_folder, R)
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'ARB_BRA_sim.png'))
    
    torch.save(gnn_model.state_dict(), os.path.join(save_path, f"gat_{R}"))

    # Training finished, start generating
    s = 0
    relation_base_dir = save_path
    prompt_pair = [ARB, BRA]
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
        "--lr",
        type=float,
        default=3e-4,
        help="The training learning rate",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="./Dataset/contain/",
        help="Path to your dataset",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="./generation_result",
        help="Path to save your checkpoints and results",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The training epochs of RRNet",
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
     train(args)





