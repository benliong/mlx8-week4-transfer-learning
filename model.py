import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPModel, CLIPProcessor, AutoTokenizer, PreTrainedModel
from utils import get_device, setup_logging
from PIL import Image   
import logging
from configuration import VisionLanguageConfig

setup_logging()
logger = logging.getLogger(__name__)

class ClipEncoder(nn.Module):
    #   Input Shape: (batch_size, 3, 224, 224)
    #   Tensor of shape (batch_size, 512)
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", freeze_clip=True):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
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
    def __init__(self, qwen_model_name="Qwen/Qwen3-0.6B-Base", freeze_qwen=True):
        super().__init__()
        self.qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name, torch_dtype=torch.bfloat16)
        if freeze_qwen:
            for param in self.qwen_model.parameters():
                param.requires_grad = False

    def forward(self, input_embeddings, labels):
        return self.qwen_model(inputs_embeds=input_embeddings, labels=labels)

class VisionLanguageModel(PreTrainedModel):
    config_class = VisionLanguageConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Initialize components with config parameters
        self.clip_encoder = ClipEncoder(
            clip_model_name=config.clip_model_name,
            freeze_clip=config.freeze_clip
        )
        self.mlp = nn.Linear(config.image_embedding_dim, config.text_embedding_dim)
        self.qwen_decoder = QwenDecoder(
            qwen_model_name=config.qwen_model_name,
            freeze_qwen=config.freeze_qwen
        )
        self.qwen_decoder.qwen_model.model.embed_tokens = self.qwen_decoder.qwen_model.model.embed_tokens.to(get_device())
        logger.info(f"[QwenDecoder] embed_tokens device: {self.qwen_decoder.qwen_model.model.embed_tokens.weight.device}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.add_special_tokens({
            "bos_token": config.bos_token,
            "eos_token": config.eos_token,
            "pad_token": config.pad_token,
            "unk_token": config.unk_token,
        })
        logger.info(f"✅ Created tokenizer with BOS token ({self.tokenizer.bos_token_id})")
        
        # Resize token embeddings
        self.qwen_decoder.qwen_model.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, images, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(get_device())
        attention_mask = attention_mask.to(get_device())
        
        # 1. Encode Images using CLIP into Patch Embeddings
        images_embeddings = self.clip_encoder(images)
        
        # 2. Adapt the image features to the Qwen model
        images_embeddings = self.mlp(images_embeddings) # map from 512 to 1024
        images_embeddings = images_embeddings.unsqueeze(1)

        # 3. Generate input for Qwen Decoder
        # 3.1 turn input_ids into text embeddings (using Qwen Tokenizer)
        input_embeddings = self.qwen_decoder.qwen_model.model.embed_tokens(input_ids)
        
        # 3.2 insert image embeddings to the front of the text embeddings
        input_embeddings = torch.cat([images_embeddings, input_embeddings], dim=1)
            
        # 3.3 generate labels from input_ids
        if labels is None:
            labels = input_ids.clone()
        
        if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        visual_ignore = torch.full((labels.size(0), 1), -100).to(labels.device)
        labels = torch.cat([visual_ignore, labels], dim=1)

        # 4. Decode the image features using the Qwen model
        qwen_output = self.qwen_decoder(input_embeddings=input_embeddings, labels=labels)
        return qwen_output

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model and tokenizer to a directory in HuggingFace format.
        """
        # Save the model configuration and weights
        super().save_pretrained(save_directory, **kwargs)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"✅ Model and tokenizer saved to: {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """
        Load a model from a directory in HuggingFace format.
        """
        # Load the configuration
        config = VisionLanguageConfig.from_pretrained(model_path)
        
        # Create the model
        model = cls(config)
        
        # Load the weights
        model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu'))
        
        # Load the tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"✅ Model loaded from: {model_path}")
        return model

# Legacy Model class for backward compatibility
class Model(VisionLanguageModel):
    def __init__(self, tokenizer_name="Qwen/Qwen3-0.6B-Base", saved_tokenizer=None):
        # Create a config for legacy compatibility
        config = VisionLanguageConfig(tokenizer_name=tokenizer_name)
        
        # Call parent constructor
        super().__init__(config)
        
        # Handle legacy tokenizer parameter
        if saved_tokenizer is not None:
            self.tokenizer = saved_tokenizer
            logger.info(f"✅ Using saved tokenizer (vocab size: {len(self.tokenizer)}, BOS ID: {self.tokenizer.bos_token_id})")
        elif tokenizer_name is not None:
            # Already initialized in parent constructor
            pass
        else:
            # No tokenizer provided - will be set later
            self.tokenizer = None
            logger.warning("⚠️ Model created without tokenizer - must be set manually")
            
        if self.tokenizer is not None:
            self.qwen_decoder.qwen_model.model.resize_token_embeddings(len(self.tokenizer))
