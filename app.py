import gradio as gr
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

prompt_tips = '''
The \<A\> and \<B\> are the first and second entities appeared in the prompt respectively. The \<R\> is the relationship between them.
Here are some example prompts for you to play with:
<pre>
1.                                             2. 
prompt: A bottle contains a car                prompt: A book is placed on the bowl
A: bottle                                      A: book
R: contains                                    R: placed
B: car                                         B: bowl

3.                                             4. 
prompt: An astronaut carries horse             prompt: A book is placed on the bowl
A: astronaut                                   A: book
R: carries                                     R: placed
B: horse                                       B: bowl
</pre>
'''

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
stable, tokenizer, noise_scheduler, vae, unet, text_encoder, image_processor = load_models(device)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.text_model.requires_grad_(False)
gnn_model = RGAT(1024, 512, 1024, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(device)
gnn_model.requires_grad_(False)
embeddings = text_encoder.get_input_embeddings()
print("load success")



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
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # stable, tokenizer, noise_scheduler, vae, unet, text_encoder, image_processor = load_models(device)
    # vae.requires_grad_(False)
    # unet.requires_grad_(False)
    # text_encoder.text_model.requires_grad_(False)
    # gnn_model = RGAT(1024, 512, 1024, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(device)
    # gnn_model.requires_grad_(False)
    # embeddings = text_encoder.get_input_embeddings()
    print(A, R, B)

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
        img_ori = stable(prompt, guidance_scale=guidance_scale, generator=torch.Generator(device).manual_seed(seed), num_inference_steps=ddim_steps, num_images_per_prompt=num_images_per_prompt).images

        delta_norm = gnn_model(h_graph_ARB, node_features_ARB)['eot'][1]
        ARB_sentence_embedding[0, ARB_eot_pos] += delta_norm * w

        prompt_embeds = torch.concat([ARB_sentence_embedding])

        stable.text_encoder = text_encoder

        img_adjusted = stable(None, guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, generator=torch.Generator(device).manual_seed(seed), num_inference_steps=ddim_steps, num_images_per_prompt=num_images_per_prompt).images

        image_grid_ori = make_image_grid(img_ori, rows=2, cols=2)
        image_grid_adjust = make_image_grid(img_adjusted, rows=2, cols=2)
        return image_grid_ori, image_grid_adjust


def check_inputs(A, B):
    input_string = A.strip()
    if " " in input_string:
        raise gr.Error(f'Expect <A> with one word as input, but received {A}.')
    input_string = B.strip()
    if " " in input_string:
        raise gr.Error(f'Expect <B> with one word as input, but received {B}.')
    
#     if control_image is None:
#         raise gr.Error("Please select or upload an Input Illusion")
#     if prompt is None or prompt == "":
#         raise gr.Error("Prompt is required")
    
def check_input_format(words):
    print("triger checking")
    input_string = words.strip()
    if " " in input_string:
        gr.Warning(f'Expect only one word, but received {words}.')
        return input_string.split()[0]
    else:
        return words
    
    
def create_view():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                sd_generation = gr.Image(
                    label='Original Stable Diffusion Output',
                    type='pil',
                    interactive=False
                )
                gr.Markdown(prompt_tips)
                prompt = gr.Textbox(
                    label='Prompt',
                    max_lines=1,
                    placeholder='A bottle contains a car')
                A = gr.Textbox(
                    label='<A>',
                    max_lines=1,
                    placeholder='bottle')
                R = gr.Dropdown(
                    label='<R>',
                    choices=["contains","carries","placed"],
                    value=0
                    )
                B = gr.Textbox(
                    label='<B>',
                    max_lines=1,
                    placeholder='car')
                
#                 A.input(check_input_format, A, A)
#                 B.input(check_input_format, B, B)



                run_button = gr.Button('Generate')

            with gr.Column():
                result = gr.Image(label='Result', interactive=False)
                adjustment_scale = gr.Slider(label='Scale of relationship adjustment',
                                               minimum=0,
                                               maximum=1,
                                               step=0.05,
                                               value=0.6)
                guidance_scale = gr.Slider(label='Classifier-Free Guidance Scale',
                                               minimum=0,
                                               maximum=50,
                                               step=0.1,
                                               value=7.5)
                ddim_steps = gr.Slider(label='Number of DDIM Sampling Steps',
                                               minimum=10,
                                               maximum=100,
                                               step=1,
                                               value=20)


        prompt.submit(
            fn=check_inputs,
            inputs=[
                A,
                B,
            ],
            queue=False
        ).success(
            fn=generate_fn,
            inputs=[
                prompt,
                A,
                R,
                B,
                guidance_scale,
                adjustment_scale,
                ddim_steps
            ],
            outputs=[sd_generation, result],
            queue=False
        )

        run_button.click(
            fn=check_inputs,
            inputs=[
                A,
                B,
            ],
            queue=False
        ).success(
            fn=generate_fn,
            inputs=[
                prompt,
                A,
                R,
                B,
                guidance_scale,
                adjustment_scale,
                ddim_steps
            ],
            outputs=[sd_generation, result],
            queue=False
        )
    return demo

TITLE = '# RRNET'
DESCRIPTION = '''
This is a gradio demo for **Relation Rectification in Diffusion Model**
'''
# DETAILDESCRIPTION='''
# RRNET
# '''
# DETAILDESCRIPTION='''
# We propose a new task, **Relation Inversion**: Given a few exemplar images, where a relation co-exists in every image, we aim to find a relation prompt **\<R>** to capture this interaction, and apply the relation to new entities to synthesize new scenes.
# Here we give several pre-trained relation prompts for you to play with. You can choose a set of exemplar images from the examples, and use **\<R>** in your prompt for relation-specific text-to-image generation.
# '''
with gr.Blocks(css='style.css') as demo:
    # if not torch.cuda.is_available():
    #     show_warning(CUDA_NOT_AVAILABLE_WARNING)

    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    # gr.Markdown(DETAILDESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('Relation-Specific Text-to-Image Generation'):
            create_view()

demo.queue().launch(share=True, server_port=6006)
