import os

# # Suppress verbose logging from tn/WeTextProcessing
# # The tn library creates loggers with INFO level and adds handlers each time,
# # so we use a filter that can't be overridden
# class _BlockAllFilter(logging.Filter):
#     def filter(self, record):
#         return False
# # Pre-configure wetext loggers BEFORE any tn imports
# for _name in ["wetext-zh_normalizer", "wetext-en_normalizer"]:
#     _logger = logging.getLogger(_name)
#     _logger.addFilter(_BlockAllFilter())
#     _logger.propagate = False  # Prevent propagation to root logger
from indextts.utils.front import TextNormalizer, TextTokenizer
from transformers import PreTrainedTokenizer
from vllm.logger import init_logger

logger = init_logger(__name__)


class IndexTTS2Tokenizer(PreTrainedTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        vocab_file = kwargs.pop("vocab_file", None)
        if vocab_file is None:
            vocab_file = os.path.join(pretrained_model_name_or_path, "bpe.model")
        logger.info(f"IndexTTS2Tokenizer.from_pretrained: {pretrained_model_name_or_path}")
        return cls(vocab_file, **kwargs)

    def __init__(self, vocab_file: str, **kwargs):
        logger.info(f"IndexTTS2Tokenizer.__init__: vocab_file={vocab_file}")
        self.vocab_file = vocab_file
        enable_glossary = kwargs.pop("enable_glossary", False)
        normalizer = TextNormalizer(enable_glossary=enable_glossary)
        self._tok = TextTokenizer(vocab_file, normalizer=normalizer)
        logger.info("IndexTTS2Tokenizer initialized successfully")
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self._tok.vocab_size

    @property
    def max_token_id(self):
        return self.vocab_size - 1

    def get_vocab(self):
        return self._tok.get_vocab()

    def _tokenize(self, text):
        return self._tok.tokenize(text)

    def _convert_token_to_id(self, token):
        return self._tok.convert_tokens_to_ids(token)[0]

    def convert_tokens_to_string(self, tokens):
        return self._tok.decode(self._tok.convert_tokens_to_ids(tokens))
