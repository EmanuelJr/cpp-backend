import logging
import json
from typing import Dict, Optional, Union

import llama_cpp
import llama_cpp.llama_tokenizer as llama_tokenizer
import llama_cpp.llama_speculative as llama_speculative
import whisper_cpp_python

from cpp_backend.schemas.model import LlamaModelSettings, WhisperModelSettings

logger = logging.getLogger(__name__)


def logit_bias_tokens_to_input_ids(
    llama: llama_cpp.Llama,
    logit_bias: Dict[str, float],
) -> Dict[str, float]:
    to_bias: Dict[str, float] = {}

    for token, score in logit_bias.items():
        token = token.encode("utf-8")
        for input_id in llama.tokenize(token, add_bos=False, special=True):
            to_bias[str(input_id)] = score

    return to_bias


_chat_format_classes = {
    "llava-1-5": llama_cpp.llama_chat_format.Llava15ChatHandler,
    "obsidian": llama_cpp.llama_chat_format.ObsidianChatHandler,
    "llava-1-6": llama_cpp.llama_chat_format.Llava16ChatHandler,
    "moondream": llama_cpp.llama_chat_format.MoondreamChatHandler,
    "nanollava": llama_cpp.llama_chat_format.NanoLlavaChatHandler,
    "llama-3-vision-alpha": llama_cpp.llama_chat_format.Llama3VisionAlpha,
    "minicpm-v-2.6": llama_cpp.llama_chat_format.MiniCPMv26ChatHandler,
}


class LlamaModelManager:
    def __init__(self):
        self.models: dict[str, llama_cpp.Llama] = {}

    def get(self, alias: str) -> Optional[llama_cpp.Llama]:
        return self.models[alias]

    def list(self):
        return list(self.models.keys())

    def load(self, model: LlamaModelSettings):
        if model.alias in self.models:
            logger.info(f"Model with alias {model.alias} already loaded")

        self.models[model.alias] = self.load_llama_from_model_settings(model)

    def unload(self, alias: str):
        logger.info(f"Unloading model with alias {alias}")
        del self.models[alias]

    @staticmethod
    def load_llama_from_model_settings(settings: LlamaModelSettings) -> llama_cpp.Llama:
        chat_handler = None
        if settings.chat_format in _chat_format_classes:
            assert settings.clip_model_path is not None, "clip model not found"
            chat_handler_class = _chat_format_classes[settings.chat_format]

            if settings.hf_model_repo_id is not None:
                chat_handler = chat_handler_class.from_pretrained(
                    repo_id=settings.hf_model_repo_id,
                    filename=settings.clip_model_path,
                    verbose=settings.verbose,
                )
            else:
                chat_handler = chat_handler_class(
                    clip_model_path=settings.clip_model_path,
                    verbose=settings.verbose,
                )
        elif settings.chat_format == "hf-autotokenizer":
            assert (
                settings.hf_pretrained_model_name_or_path is not None
            ), "hf_pretrained_model_name_or_path must be set for hf-autotokenizer"

            chat_handler = (
                llama_cpp.llama_chat_format.hf_autotokenizer_to_chat_completion_handler(
                    settings.hf_pretrained_model_name_or_path
                )
            )
        elif settings.chat_format == "hf-tokenizer-config":
            assert (
                settings.hf_tokenizer_config_path is not None
            ), "hf_tokenizer_config_path must be set for hf-tokenizer-config"

            chat_handler = llama_cpp.llama_chat_format.hf_tokenizer_config_to_chat_completion_handler(
                json.load(open(settings.hf_tokenizer_config_path))
            )

        tokenizer: Optional[llama_cpp.BaseLlamaTokenizer] = None
        if settings.hf_pretrained_model_name_or_path is not None:
            tokenizer = llama_tokenizer.LlamaHFTokenizer.from_pretrained(
                settings.hf_pretrained_model_name_or_path
            )

        draft_model = None
        if settings.draft_model is not None:
            draft_model = llama_speculative.LlamaPromptLookupDecoding(
                num_pred_tokens=settings.draft_model_num_pred_tokens
            )

        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None
        if settings.kv_overrides is not None:
            assert isinstance(settings.kv_overrides, list)
            kv_overrides = {}
            for kv in settings.kv_overrides:
                key, value = kv.split("=")
                if ":" in value:
                    value_type, value = value.split(":")
                    if value_type == "bool":
                        kv_overrides[key] = value.lower() in ["true", "1"]
                    elif value_type == "int":
                        kv_overrides[key] = int(value)
                    elif value_type == "float":
                        kv_overrides[key] = float(value)
                    elif value_type == "str":
                        kv_overrides[key] = value
                    else:
                        raise ValueError(f"Unknown value type {value_type}")

        import functools

        kwargs = {}

        if settings.hf_model_repo_id is not None:
            create_fn = functools.partial(
                llama_cpp.Llama.from_pretrained,
                repo_id=settings.hf_model_repo_id,
                filename=settings.model,
            )
        else:
            create_fn = llama_cpp.Llama
            kwargs["model_path"] = settings.model

        _model = create_fn(
            **kwargs,
            # Model Params
            n_gpu_layers=settings.n_gpu_layers,
            split_mode=settings.split_mode,
            main_gpu=settings.main_gpu,
            tensor_split=settings.tensor_split,
            vocab_only=settings.vocab_only,
            use_mmap=settings.use_mmap,
            use_mlock=settings.use_mlock,
            kv_overrides=kv_overrides,
            rpc_servers=settings.rpc_servers,
            # Context Params
            seed=settings.seed,
            n_ctx=settings.n_ctx,
            n_batch=settings.n_batch,
            n_ubatch=settings.n_ubatch,
            n_threads=settings.n_threads,
            n_threads_batch=settings.n_threads_batch,
            rope_scaling_type=settings.rope_scaling_type,
            rope_freq_base=settings.rope_freq_base,
            rope_freq_scale=settings.rope_freq_scale,
            yarn_ext_factor=settings.yarn_ext_factor,
            yarn_attn_factor=settings.yarn_attn_factor,
            yarn_beta_fast=settings.yarn_beta_fast,
            yarn_beta_slow=settings.yarn_beta_slow,
            yarn_orig_ctx=settings.yarn_orig_ctx,
            mul_mat_q=settings.mul_mat_q,
            logits_all=settings.logits_all,
            embedding=settings.embedding,
            offload_kqv=settings.offload_kqv,
            flash_attn=settings.flash_attn,
            # Sampling Params
            last_n_tokens_size=settings.last_n_tokens_size,
            # LoRA Params
            lora_base=settings.lora_base,
            lora_path=settings.lora_path,
            # Backend Params
            numa=settings.numa,
            # Chat Format Params
            chat_format=settings.chat_format,
            chat_handler=chat_handler,
            # Speculative Decoding
            draft_model=draft_model,
            # KV Cache Quantization
            type_k=settings.type_k,
            type_v=settings.type_v,
            # Tokenizer
            tokenizer=tokenizer,
            # Misc
            verbose=settings.verbose,
        )
        if settings.cache:
            if settings.cache_type == "disk":
                if settings.verbose:
                    print(f"Using disk cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
            else:
                if settings.verbose:
                    print(f"Using ram cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)
            _model.set_cache(cache)
        return _model


class WhisperModelManager:
    def __init__(self):
        self.models: dict[str, whisper_cpp_python.Whisper] = {}

    def get(self, alias: str) -> Optional[whisper_cpp_python.Whisper]:
        return self.models[alias]

    def list(self):
        return list(self.models.keys())

    def load(self, model: WhisperModelSettings):
        if model.alias in self.models:
            logger.info(f"Model with alias {model.alias} already loaded")

        settings = model.model_dump(exclude=["alias", "model"])
        self.models[model.alias] = whisper_cpp_python.Whisper(
            model_path=model.model, **settings
        )

    def unload(self, alias: str):
        logger.info(f"Unloading model with alias {alias}")
        del self.models[alias]


# Probably we will have some concurrency issues here in special
llama_models = LlamaModelManager()
whisper_models = WhisperModelManager()
