from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers import TokenizerRegistry

TokenizerRegistry.register(
    "indextts2",
    module="vllm_omni.model_executor.models.indextts2.tokenizer",
    class_name="IndexTTS2Tokenizer",
)

RENDERER_REGISTRY.register(
    "indextts2",
    module="vllm_omni.tokenizers.indextts2_renderer",
    class_name="IndexTTS2Renderer",
)
