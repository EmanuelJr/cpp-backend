import multiprocessing
from typing import Optional, List, Literal, TypedDict, Union
from pydantic import BaseModel, Field
import llama_cpp


class LlamaModelSettings(BaseModel):
    """Model settings used to load a Llama model."""

    alias: str = Field(
        description="Alias of the model used for generating completions."
    )
    model: str = Field(description="Path to the model used for generating completions.")
    # Model Params
    n_gpu_layers: int = Field(
        default=0,
        description=(
            "Number of layers to place on the GPU; remaining layers will be on the CPU. "
            "Set to -1 to move all layers to the GPU."
        ),
        ge=-1,
    )
    split_mode: int = Field(
        default=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        description="Method used to split layers between CPU and GPU.",
    )
    main_gpu: int = Field(
        default=0,
        description="Index of the main GPU to use.",
        ge=0,
    )
    tensor_split: Optional[List[float]] = Field(
        default=None,
        description="List of proportions to split layers across multiple GPUs.",
    )
    vocab_only: bool = Field(
        default=False, description="If True, only return the vocabulary."
    )
    use_mmap: bool = Field(
        default=llama_cpp.llama_supports_mmap(),
        description="Whether to use memory-mapped files (mmap).",
    )
    use_mlock: bool = Field(
        default=llama_cpp.llama_supports_mlock(),
        description="Whether to use memory locking (mlock).",
    )
    kv_overrides: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of model key-value overrides in the format 'key=type:value', "
            "where type is one of 'bool', 'int', or 'float'. Valid true values are 'true', 'TRUE', or '1'; otherwise false."
        ),
    )
    rpc_servers: Optional[str] = Field(
        default=None, description="Comma-separated list of RPC servers for offloading."
    )
    # Context Params
    seed: int = Field(
        default=llama_cpp.LLAMA_DEFAULT_SEED,
        description="Random seed for reproducibility; set to -1 for a random seed.",
    )
    n_ctx: int = Field(
        default=2048, ge=0, description="Context size (number of tokens)."
    )
    n_batch: int = Field(
        default=512, ge=1, description="Batch size to use per evaluation."
    )
    n_ubatch: int = Field(
        default=512, ge=1, description="Physical batch size used by llama.cpp."
    )
    n_threads: int = Field(
        default=1,
        ge=1,
        description="Number of threads to use.",
    )
    n_threads_batch: int = Field(
        default=max(multiprocessing.cpu_count(), 1),
        ge=0,
        description="Number of threads to use when batch processing.",
    )
    rope_scaling_type: int = Field(
        default=llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        description="Type of RoPE (Rotary Position Embedding) scaling to use.",
    )
    rope_freq_base: float = Field(
        default=0.0, description="Base frequency for RoPE (Rotary Position Embedding)."
    )
    rope_freq_scale: float = Field(
        default=0.0,
        description="Frequency scaling factor for RoPE (Rotary Position Embedding).",
    )
    yarn_ext_factor: float = Field(
        default=-1.0, description="Extension factor for YARN."
    )
    yarn_attn_factor: float = Field(
        default=1.0, description="Attention factor for YARN."
    )
    yarn_beta_fast: float = Field(
        default=32.0, description="Fast beta parameter for YARN."
    )
    yarn_beta_slow: float = Field(
        default=1.0, description="Slow beta parameter for YARN."
    )
    yarn_orig_ctx: int = Field(default=0, description="Original context size for YARN.")
    mul_mat_q: bool = Field(
        default=True, description="If True, use experimental mul_mat_q kernels."
    )
    logits_all: bool = Field(
        default=True, description="If True, return logits for each token."
    )
    embedding: bool = Field(
        default=False, description="If True, compute and return embeddings."
    )
    offload_kqv: bool = Field(
        default=True,
        description="If True, offload key, query, and value matrices (kqv) to the GPU.",
    )
    flash_attn: bool = Field(default=False, description="If True, use flash attention.")
    # Sampling Params
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description="Number of recent tokens to consider for repeat penalty calculation.",
    )
    # LoRA Params
    lora_base: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to a base model; useful when applying LoRA to an f16 model "
            "while using a quantized base model."
        ),
    )
    lora_path: Optional[str] = Field(
        default=None,
        description="Path to a LoRA (Low-Rank Adaptation) file to apply to the model.",
    )
    # Backend Params
    numa: Union[bool, int] = Field(
        default=False, description="Enable NUMA (Non-Uniform Memory Access) support."
    )
    # Chat Format Params
    chat_format: Optional[str] = Field(
        default=None, description="Chat format to use for conversations."
    )
    clip_model_path: Optional[str] = Field(
        default=None,
        description="Path to a CLIP (Contrastive Language–Image Pretraining) model for multi-modal chat completion.",
    )
    # Cache Params
    cache: bool = Field(
        default=False,
        description="If True, use a cache to reduce processing times for evaluated prompts.",
    )
    cache_type: Literal["ram", "disk"] = Field(
        default="ram",
        description="Type of cache to use ('ram' or 'disk'); only used if cache is True.",
    )
    cache_size: int = Field(
        default=2 << 30,
        description="Size of the cache in bytes; only used if cache is True.",
    )
    # Tokenizer Options
    hf_tokenizer_config_path: Optional[str] = Field(
        default=None, description="Path to a HuggingFace 'tokenizer_config.json' file."
    )
    hf_pretrained_model_name_or_path: Optional[str] = Field(
        default=None,
        description=(
            "Model name or path to a pretrained HuggingFace tokenizer model, "
            "as you would pass to 'AutoTokenizer.from_pretrained()'."
        ),
    )
    # Loading from HuggingFace Model Hub
    hf_model_repo_id: Optional[str] = Field(
        default=None,
        description="Model repository ID for the HuggingFace tokenizer model.",
    )
    # Speculative Decoding
    draft_model: Optional[str] = Field(
        default=None,
        description=(
            "Method to use for speculative decoding; one of 'prompt', 'lookup', or 'decoding'."
        ),
    )
    draft_model_num_pred_tokens: int = Field(
        default=10, description="Number of tokens to predict using the draft model."
    )
    # KV Cache Quantization
    type_k: Optional[int] = Field(
        default=None, description="Data type for quantization of the key cache."
    )
    type_v: Optional[int] = Field(
        default=None, description="Data type for quantization of the value cache."
    )
    # Misc
    verbose: bool = Field(default=True, description="If True, print debug information.")


class WhisperModelSettings(BaseModel):
    """Model settings used to load a Whisper model."""

    alias: str = Field(description="Alias of the model used for audio processing.")
    model: str = Field(description="Path to the model used for audio processing.")
    strategy: int = Field(
        default=0, description="Strategy to use for audio processing."
    )
    n_threads: int = Field(default=1, ge=1, description="Number of threads to use.")


class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]


class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]


class ModelUnloadRequest(BaseModel):
    alias: str = Field(
        description="The alias of the model to used on loading.",
    )
