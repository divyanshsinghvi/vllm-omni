import os

from indextts.utils.front import TextNormalizer, TextTokenizer
from transformers import PreTrainedTokenizer


class IndexTTS2Tokenizer(PreTrainedTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        vocab_file = kwargs.pop("vocab_file", None)
        if vocab_file is None:
            vocab_file = os.path.join(pretrained_model_name_or_path, "bpe.model")
        return cls(vocab_file, **kwargs)

    def __init__(self, vocab_file: str, **kwargs):
        self.vocab_file = vocab_file
        enable_glossary = kwargs.pop("enable_glossary", False)
        normalizer = TextNormalizer(enable_glossary=enable_glossary)
        self._tok = TextTokenizer(vocab_file, normalizer=normalizer)
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self._tok.vocab_size

    def get_vocab(self):
        return self._tok.get_vocab()

    def _tokenize(self, text):
        return self._tok.tokenize(text)

    def _convert_token_to_id(self, token):
        return self._tok.convert_tokens_to_ids(token)[0]

    def convert_tokens_to_string(self, tokens):
        return self._tok.decode(self._tok.convert_tokens_to_ids(tokens))
