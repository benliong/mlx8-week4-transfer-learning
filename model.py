import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPModel, CLIPProcessor, AutoTokenizer
from utils import get_device
from PIL import Image   
import logging

class ClipEncoder(nn.Module):
    #   Input Shape: (batch_size, 3, 224, 224)
    #   Tensor of shape (batch_size, 512)
    def __init__(self, freeze_clip=True):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        images = self.processor(images=images, return_tensors="pt")["pixel_values"]
        images = images.to(get_device())
        return self.clip_model.get_image_features(images)

class QwenDecoder(nn.Module):
    # Qwen Decoder Output:
    #   CausalLMOutputWithPast(
    #     loss=Tensor of shape (),  # scalar loss
    #     logits=Tensor of shape (batch_size, seq_len, vocab_size),
    #     ...
    #   )
    def __init__(self, freeze_qwen=True):
        super().__init__()
        self.qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", device_map=get_device(), torch_dtype=torch.bfloat16)
        if freeze_qwen:
            for param in self.qwen_model.parameters():
                param.requires_grad = False

    def forward(self, input_embeddings, labels):
        return self.qwen_model(inputs_embeds=input_embeddings, labels=labels)

class Model(nn.Module):
    def __init__(self, tokenizer_name="Qwen/Qwen3-0.6B-Base"):
        super().__init__()
        self.clip_encoder = ClipEncoder()
        self.mlp = nn.Linear(512, 1024)
        self.qwen_decoder = QwenDecoder()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def forward(self, images, input_ids, attention_mask):

        print("images shape: PIL")
        print("input_ids shape: ", input_ids.shape)
        print("attention_mask shape: ", attention_mask.shape)

        # 1. Encode Images using CLIP into Patch Embeddings
        print("[Clip_encoder]")
        images_embeddings = self.clip_encoder(images)
        print("images_embeddings shape: ", images_embeddings.shape)
        print("[/Clip_encoder]")

        # 2. Adapt the image features to the Qwen model
        print("[MLP]")
        images_embeddings = self.mlp(images_embeddings) # map from 512 to 1024
        print("image embeddings shape: ", images_embeddings.shape)
        print("[/MLP]")

        print("[unsqueeze]")
        images_embeddings = images_embeddings.unsqueeze(1)
        print("images_embeddings shape: ", images_embeddings.shape)
        print("[/unsqueeze]")

        # 3 Generate input for Qwen Decoder // dimension == 1024
        # 3.1 turn input_ids into text embeddings (using Qwen Tokenizer)
        input_embeddings = self.qwen_decoder.qwen_model.model.embed_tokens(input_ids) # "A man with a beard" -> [15919, 525, 19837, 151643, 151643, 151643]
        # 3.2 insert image embeddings to the front of the text embeddings
        print("[Concat]...")
        print("image embeddings shape: ", images_embeddings.shape)
        print("input embeddings shape: ", input_embeddings.shape)
        input_embeddings = torch.cat([images_embeddings, input_embeddings], dim=1) # [Image, A, Man, with, a, beard]
        print("input embeddings shape: ", input_embeddings.shape)
            
        # input_embeddings.input_ids = input_embeddings.input_ids
        # 3.3 generate labels from input_ids
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        visual_ignore = torch.full((labels.size(0), 1), -100).to(labels.device)
        labels = torch.cat([visual_ignore, labels], dim=1)  # shape: [B, T+1]
        labels = labels

        # 4. Decode the image features using the Qwen model
        # raw caption = "A Man with a beard"
        # tokenized caption = [15919, 525, 19837, 151643, 151643, 151643]
        # input_embeddings = [1024, 1024, 1024, 1024, 1024, 1024]
        # labels = [1024, 1024, 1024, 1024, 1024, 1024]
        # labels = [1024, 1024, 1024, 1024, 1024, 1024]


        # input Embeddings:     [Image, <bos>, A, Man, with, a, beard] 
        # labels / Mask:        [-100, 1, -100, -100, -100, -100, -100, -100]

        #input                  [<bos>, the ]
        # loss = the (result) - a (expected output) [Cross Entropy Loss]
        #         
        qwen_output = self.qwen_decoder(input_embeddings=input_embeddings, labels=labels)
        return qwen_output
