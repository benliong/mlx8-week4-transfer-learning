from transformers import PretrainedConfig

class VisionLanguageConfig(PretrainedConfig):
    model_type = "vision_language_model"
    
    def __init__(
        self,
        # Model architecture parameters
        clip_model_name="openai/clip-vit-base-patch32",
        qwen_model_name="Qwen/Qwen3-0.6B-Base",
        image_embedding_dim=512,
        text_embedding_dim=1024,
        freeze_clip=True,
        freeze_qwen=True,
        
        # Training parameters
        max_caption_length=128,
        image_size=224,
        
        # Tokenizer parameters
        tokenizer_name="Qwen/Qwen3-0.6B-Base",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        
        **kwargs,
    ):
        self.clip_model_name = clip_model_name
        self.qwen_model_name = qwen_model_name
        self.image_embedding_dim = image_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.freeze_clip = freeze_clip
        self.freeze_qwen = freeze_qwen
        
        self.max_caption_length = max_caption_length
        self.image_size = image_size
        
        self.tokenizer_name = tokenizer_name
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        super().__init__(**kwargs)