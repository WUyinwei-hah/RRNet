import torch
from GNN import create_hg_A_ARB, create_hg_A_ARB_self_loop
from torch.utils.data import Dataset
import random
import os
from PIL import Image

class HeteGraphDataset:
    def __init__(self, base_path, A, B, R, ARB, BRA, ARB_dir, BRA_dir, image_processor, tokenizer, transform=None):

        self.A = A
        self.B = B
        self.R = R
        self.ARB = ARB
        self.BRA = BRA
        
        self.A_directory = os.path.join(base_path, A)
        self.B_directory = os.path.join(base_path, B)
        self.ARB_directory = os.path.join(base_path, ARB_dir)
        self.BRA_directory = os.path.join(base_path, BRA_dir)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        self.A_imgs = [image_processor.preprocess(Image.open(os.path.join(self.A_directory, f)).convert("RGB"), height=512, width=512) for f in os.listdir(self.A_directory) if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]
        self.B_imgs = [image_processor.preprocess(Image.open(os.path.join(self.B_directory, f)).convert("RGB"), height=512, width=512) for f in os.listdir(self.B_directory) if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]
        self.ARB_imgs = [image_processor.preprocess(Image.open(os.path.join(self.ARB_directory, f)).convert("RGB"), height=512, width=512) for f in os.listdir(self.ARB_directory) if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]
        self.BRA_imgs = [image_processor.preprocess(Image.open(os.path.join(self.BRA_directory, f)).convert("RGB"), height=512, width=512) for f in os.listdir(self.BRA_directory) if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]
        
        self.transform = transform
    
    def get_eot_postiton(self, sentence):
        untruncated_ids = self.tokenizer([sentence], padding="longest", return_tensors="pt").input_ids
        eot_pos = len(untruncated_ids[0])-1
        return eot_pos
        
    def get_word_embeddings(self, embedding, word, device):

        txt_id = self.tokenizer(
                [word],
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
        word_embedding = embedding(txt_id[0, 1])
        word_embedding = word_embedding.unsqueeze(0)
        return word_embedding
    
    def get_sentence_eot_embeddings(self, text_encoder, sentence, device):

        txt_id = self.tokenizer(
                [sentence],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
        
        eot_pos = self.get_eot_postiton(sentence)
        sentence_embedding = text_encoder(txt_id)[0].to(device)[0]
        
        eot_embedding = sentence_embedding[eot_pos]
        eot_embedding = eot_embedding.unsqueeze(0)
        return eot_embedding, sentence_embedding.unsqueeze(0), eot_pos
    

    def sample(self, A_word, R_word, B_word, ARB_sentence, BRA_sentence, text_encoder, embeddings, device, self_loop=True, add_noise_on_A=True):
        
        def get_object_sentence(obj):
            return f"this is a photo of {obj}"
        
        A_word_embedding = self.get_word_embeddings(embeddings, A_word, device)
        R_word_embedding = self.get_word_embeddings(embeddings, R_word, device)
        B_word_embedding = self.get_word_embeddings(embeddings, B_word, device)
        
        ARB_eot, ARB_sentence_embedding, ARB_eot_pos = self.get_sentence_eot_embeddings(text_encoder, ARB_sentence, device)
        BRA_eot, BRA_sentence_embedding, BRA_eot_pos = self.get_sentence_eot_embeddings(text_encoder, BRA_sentence, device)
        
        
        A_eot_embedding, A_sentence_embedding, A_eot_pos = self.get_sentence_eot_embeddings(text_encoder, get_object_sentence(A_word), device)
        B_eot_embedding, B_sentence_embedding, B_eot_pos = self.get_sentence_eot_embeddings(text_encoder, get_object_sentence(B_word), device)
        
        if self_loop:
            
            h_graph_ARB, node_features_ARB = create_hg_A_ARB_self_loop(A_word_embedding, R_word_embedding, B_word_embedding, A_eot_embedding, ARB_eot, device)
            h_graph_BRA, node_features_BRA = create_hg_A_ARB_self_loop(B_word_embedding, R_word_embedding, A_word_embedding, B_eot_embedding, BRA_eot, device)
        else:
            h_graph_ARB, node_features_ARB = create_hg_A_ARB(A_word_embedding, R_word_embedding, B_word_embedding, A_eot_embedding, ARB_eot, device)
            h_graph_BRA, node_features_BRA = create_hg_A_ARB(B_word_embedding, R_word_embedding, A_word_embedding, B_eot_embedding, BRA_eot, device)
            
            
        A_img = random.choice(self.A_imgs)
        B_img = random.choice(self.B_imgs)
        ARB_img = random.choice(self.ARB_imgs)
        BRA_img = random.choice(self.BRA_imgs)
        

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
            ARB_img = self.transform(ARB_img)
            BRA_img = self.transform(BRA_img)
        

        return A_img, B_img, ARB_img, BRA_img, ARB_sentence_embedding, ARB_eot_pos, BRA_sentence_embedding, BRA_eot_pos, \
                A_eot_embedding, A_sentence_embedding, A_eot_pos, B_eot_embedding, B_sentence_embedding, B_eot_pos, \
                h_graph_ARB, node_features_ARB, h_graph_BRA, node_features_BRA
    