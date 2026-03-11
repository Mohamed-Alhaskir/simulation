"""
utils/llm_backends.py
---------------------
Thin backend abstraction used by LLMAnalysisStage.
Supports llama_cpp and vllm backends.
"""

from __future__ import annotations

import logging


class LLMBackend:
    """Abstract base for LLM backends."""

    def generate(self, prompt: str, cfg: dict) -> str:
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    def __init__(self, cfg: dict, logger_instance: logging.Logger | None = None):
        self.logger = logger_instance or logging.getLogger(__name__)
        self._model = None
        self._cfg = cfg

    def _load(self, cfg: dict) -> None:
        if self._model is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            self.logger.error(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
            raise

        model_path = cfg["model_path"]
        self.logger.info(f"Loading llama model: {model_path}")

        requested_ctx  = cfg.get("context_length", 8192)
        n_gpu_layers   = cfg.get("n_gpu_layers", -1)
        seed           = cfg.get("seed", 42)

        # Build kwargs — only add type_k/type_v/flash_attn if explicitly set
        # in config, so we don't break older llama-cpp-python versions that
        # take integer enums or don't support these params at all.
        extra = {}
        if "type_k" in cfg:
            extra["type_k"] = cfg["type_k"]
        if "type_v" in cfg:
            extra["type_v"] = cfg["type_v"]
        if "flash_attn" in cfg:
            extra["flash_attn"] = cfg["flash_attn"]

        # Try flash_attn=True first (saves ~10-15% VRAM) if not already set
        if "flash_attn" not in extra:
            extra_with_flash = dict(extra, flash_attn=True)
        else:
            extra_with_flash = extra

        # Fallback ladder: (n_ctx, n_gpu_layers, extra_kwargs)
        # Start with flash attn; if that param isn't supported, drop it.
        attempts = [
            (requested_ctx, n_gpu_layers,  extra_with_flash),
            (requested_ctx, n_gpu_layers,  extra),            # without flash_attn
            (32768,         n_gpu_layers,  extra),
            (16384,         n_gpu_layers,  extra),
            (8192,          n_gpu_layers,  extra),
            (8192,          75,            extra),            # partial GPU offload
            (4096,          75,            {}),               # bare minimum
        ]

        # Deduplicate
        seen = []
        deduped = []
        for a in attempts:
            key = (a[0], a[1], tuple(sorted(a[2].items())))
            if key not in seen:
                seen.append(key)
                deduped.append(a)

        last_exc = None
        for n_ctx, n_gpu, kwargs in deduped:
            if (n_ctx, n_gpu, kwargs) != (requested_ctx, n_gpu_layers, extra_with_flash):
                self.logger.warning(
                    f"Retrying: n_ctx={n_ctx}, n_gpu_layers={n_gpu}, "
                    f"extra={list(kwargs)}"
                )
            try:
                self._model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu,
                    seed=seed,
                    verbose=False,
                    **kwargs,
                )
                if n_ctx < requested_ctx:
                    self.logger.warning(
                        f"Model loaded with n_ctx={n_ctx} "
                        f"(requested {requested_ctx} reduced to fit in VRAM)"
                    )
                return  # success
            except Exception as e:
                last_exc = e
                self.logger.warning(f"Load attempt failed: {e}")

        raise RuntimeError(
            f"Failed to load {model_path} after all fallback attempts. "
            f"Last error: {last_exc}"
        )

    def generate(self, prompt: str, cfg: dict) -> str:
        self._load(cfg)
        response = self._model(
            prompt,
            max_tokens=cfg.get("max_tokens", 6144),
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            repeat_penalty=cfg.get("repeat_penalty", 1.05),
            stop=None,
        )
        return response["choices"][0]["text"]

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    def cleanup(self) -> None:
        self.unload()


class VLLMBackend(LLMBackend):
    def __init__(self, cfg: dict, logger_instance: logging.Logger | None = None):
        self.logger = logger_instance or logging.getLogger(__name__)
        self._model = None
        self._cfg = cfg

    def _load(self, cfg: dict) -> None:
        if self._model is not None:
            return
        try:
            from vllm import LLM
        except ImportError:
            self.logger.error("vLLM not installed. Install with: pip install vllm")
            raise
        model_name = cfg.get("model_name", cfg.get("model_path"))
        self.logger.info(f"Loading vLLM model: {model_name}")
        self._model = LLM(
            model=model_name,
            max_model_len=cfg.get("context_length", 8192),
            seed=cfg.get("seed", 42),
        )

    def generate(self, prompt: str, cfg: dict) -> str:
        self._load(cfg)
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            max_tokens=cfg.get("max_tokens", 6144),
        )
        outputs = self._model.generate([prompt], params)
        return outputs[0].outputs[0].text

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    def cleanup(self) -> None:
        self.unload()


_BACKENDS: dict[str, type[LLMBackend]] = {
    "llama_cpp": LlamaCppBackend,
    "vllm": VLLMBackend,
}


def get_llm_backend(
    backend_name: str,
    cfg: dict,
    logger_instance: logging.Logger | None = None,
) -> LLMBackend:
    cls = _BACKENDS.get(backend_name)
    if cls is None:
        raise ValueError(
            f"Unknown LLM backend: {backend_name!r}. "
            f"Available: {list(_BACKENDS)}"
        )
    return cls(cfg, logger_instance=logger_instance)