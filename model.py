import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPModel, CLIPProcessor, AutoTokenizer, AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel
from utils import get_device, setup_logging
from PIL import Image   
import logging

setup_logging()
logger = logging.getLogger(__name__)

class MultimodalClipQwenConfig(PretrainedConfig):
    """
    Configuration class for MultimodalClipQwen model.
    """
    model_type = "multimodal_clip_qwen"
    
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        qwen_model_name="Qwen/Qwen3-0.6B-Base",
        mlp_hidden_size=1024,
        clip_hidden_size=512,
        freeze_clip=True,
        freeze_qwen=True,
        tokenizer_name="Qwen/Qwen3-0.6B-Base",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.clip_model_name = clip_model_name
        self.qwen_model_name = qwen_model_name
        self.mlp_hidden_size = mlp_hidden_size
        self.clip_hidden_size = clip_hidden_size
        self.freeze_clip = freeze_clip
        self.freeze_qwen = freeze_qwen
        self.tokenizer_name = tokenizer_name

class ClipEncoder(nn.Module):
    #   Input Shape: (batch_size, 3, 224, 224)
    #   Tensor of shape (batch_size, 512)
    def __init__(self, freeze_clip=True, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        # Process images with the CLIP processor
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        pixel_values = pixel_values.to(get_device())
        return self.clip_model.get_image_features(pixel_values)

class QwenDecoder(nn.Module):
    # Qwen Decoder Output:
    #   CausalLMOutputWithPast(
    #     loss=Tensor of shape (),  # scalar loss
    #     logits=Tensor of shape (batch_size, seq_len, vocab_size),
    #     ...
    #   )
    def __init__(self, freeze_qwen=True, qwen_model_name="Qwen/Qwen3-0.6B-Base"):
        super().__init__()
        self.qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name, torch_dtype=torch.bfloat16)
        if freeze_qwen:
            for param in self.qwen_model.parameters():
                param.requires_grad = False

    def forward(self, input_embeddings, labels):
        return self.qwen_model(inputs_embeds=input_embeddings, labels=labels)

class MultimodalClipQwenModel(PreTrainedModel):
    """
    Multimodal model combining CLIP vision encoder with Qwen language model.
    Compatible with transformers AutoModel.
    """
    config_class = MultimodalClipQwenConfig
    base_model_prefix = "multimodal_clip_qwen"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Initialize components
        self.clip_encoder = ClipEncoder(
            freeze_clip=config.freeze_clip,
            clip_model_name=config.clip_model_name
        )
        self.mlp = nn.Linear(config.clip_hidden_size, config.mlp_hidden_size)
        self.qwen_decoder = QwenDecoder(
            freeze_qwen=config.freeze_qwen,
            qwen_model_name=config.qwen_model_name
        )
        
        # Move embed_tokens to the right device
        self.qwen_decoder.qwen_model.model.embed_tokens = self.qwen_decoder.qwen_model.model.embed_tokens.to(get_device())
        print("[QwenDecoder] embed_tokens device:", self.qwen_decoder.qwen_model.model.embed_tokens.weight.device)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
        logger.info(f"✅ Created fresh tokenizer with BOS token ({self.tokenizer.bos_token_id})")
        
        # Resize embeddings to match tokenizer
        self.qwen_decoder.qwen_model.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, images, input_ids, attention_mask):
        input_ids = input_ids.to(get_device())
        attention_mask = attention_mask.to(get_device())
        
        # 1. Encode Images using CLIP into Patch Embeddings
        images_embeddings = self.clip_encoder(images)
        
        # 2. Adapt the image features to the Qwen model
        images_embeddings = self.mlp(images_embeddings) # map from 512 to 1024
        
        # 3. Prepare embeddings
        images_embeddings = images_embeddings.unsqueeze(1)

        # 4. Generate input for Qwen Decoder
        input_embeddings = self.qwen_decoder.qwen_model.model.embed_tokens(input_ids)
        input_embeddings = torch.cat([images_embeddings, input_embeddings], dim=1)
            
        # 5. Generate labels from input_ids
        labels = input_ids.clone()
        if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        visual_ignore = torch.full((labels.size(0), 1), -100).to(labels.device)
        labels = torch.cat([visual_ignore, labels], dim=1)

        # 6. Decode using Qwen model
        qwen_output = self.qwen_decoder(input_embeddings=input_embeddings, labels=labels)
        return qwen_output

# Legacy Model class for backward compatibility
class Model(nn.Module):
    def __init__(self, tokenizer_name="Qwen/Qwen3-0.6B-Base", saved_tokenizer=None):
        super().__init__()
        self.clip_encoder = ClipEncoder()
        self.mlp = nn.Linear(512, 1024)
        self.qwen_decoder = QwenDecoder()
        self.qwen_decoder.qwen_model.model.embed_tokens = self.qwen_decoder.qwen_model.model.embed_tokens.to(get_device())
        print("[QwenDecoder] embed_tokens device:", self.qwen_decoder.qwen_model.model.embed_tokens.weight.device)

        if saved_tokenizer is not None:
            # Use the provided saved tokenizer (for loading saved models)
            self.tokenizer = saved_tokenizer
            logger.info(f"✅ Using saved tokenizer (vocab size: {len(self.tokenizer)}, BOS ID: {self.tokenizer.bos_token_id})")
        elif tokenizer_name is not None:
            # Create new tokenizer (for training)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
            logger.info(f"✅ Created fresh tokenizer with BOS token ({self.tokenizer.bos_token_id})")
        else:
            # No tokenizer provided - will be set later
            self.tokenizer = None
            logger.warning("⚠️ Model created without tokenizer - must be set manually")
            
        if self.tokenizer is not None:
            self.qwen_decoder.qwen_model.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, images, input_ids, attention_mask):
        input_ids = input_ids.to(get_device())
        attention_mask = attention_mask.to(get_device())

        # 1. Encode Images using CLIP into Patch Embeddings
        images_embeddings = self.clip_encoder(images)
        
        # 2. Adapt the image features to the Qwen model
        images_embeddings = self.mlp(images_embeddings) # map from 512 to 1024
        
        images_embeddings = images_embeddings.unsqueeze(1)

        # 3 Generate input for Qwen Decoder // dimension == 1024
        # 3.1 turn input_ids into text embeddings (using Qwen Tokenizer)
        input_embeddings = self.qwen_decoder.qwen_model.model.embed_tokens(input_ids) # "A man with a beard" -> [15919, 525, 19837, 151643, 151643, 151643]
        # 3.2 insert image embeddings to the front of the text embeddings
        input_embeddings = torch.cat([images_embeddings, input_embeddings], dim=1) # [Image, A, Man, with, a, beard]
            
        # 3.3 generate labels from input_ids
        labels = input_ids.clone()
        if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        visual_ignore = torch.full((labels.size(0), 1), -100).to(labels.device)
        labels = torch.cat([visual_ignore, labels], dim=1)  # shape: [B, T+1]
        labels = labels

        # 4. Decode the image features using the Qwen model
        qwen_output = self.qwen_decoder(input_embeddings=input_embeddings, labels=labels)
        return qwen_output

# Register the model with transformers (only once per process)

# Global flag to prevent duplicate registrations within the same process
_AUTOMODEL_REGISTERED = False

if not _AUTOMODEL_REGISTERED:
    AutoConfig.register("multimodal_clip_qwen", MultimodalClipQwenConfig)
    AutoModel.register(MultimodalClipQwenConfig, MultimodalClipQwenModel)
    logger.info("✅ MultimodalClipQwen model registered with transformers AutoModel")
    _AUTOMODEL_REGISTERED = True
