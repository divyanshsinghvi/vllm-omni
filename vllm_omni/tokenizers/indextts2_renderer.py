# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

from vllm.config import VllmConfig
from vllm.renderers.hf import HfRenderer, HfTokenizer
from vllm.tokenizers import TokenizerRegistry


class IndexTTS2Renderer(HfRenderer):
    """Renderer that loads the IndexTTS2 BPE tokenizer via the registry
    instead of forcing ``AutoTokenizer``."""

    @classmethod
    def from_config(cls, config: VllmConfig, tokenizer_kwargs: dict[str, Any]) -> "IndexTTS2Renderer":
        model_config = config.model_config
        if model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer_cls = TokenizerRegistry.load_tokenizer_cls("indextts2")
            tokenizer_name = tokenizer_kwargs.pop("tokenizer_name")
            tokenizer = cast(
                HfTokenizer,
                tokenizer_cls.from_pretrained(tokenizer_name, **tokenizer_kwargs),
            )
        return cls(config, tokenizer)
