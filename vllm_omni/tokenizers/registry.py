from vllm.tokenizers import TokenizerRegistry

TokenizerRegistry.register(
    "indextts2",
    module="vllm_omni.model_executor.models.indextts2.tokenizer",
    class_name="IndexTTS2Tokenizer",
)
