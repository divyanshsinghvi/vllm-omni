from pathlib import Path

import yaml
from vllm.logger import init_logger
from vllm.transformers_utils.config import register_config_parser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from vllm_omni.model_executor.models.indextts2.index_tts_config import (
    IndexTTS2Config,
)

logger = init_logger(__name__)


@register_config_parser("indextts2_yaml")
class IndexTTS2YamlConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ):
        config_path = Path(model) / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"IndexTTS2 config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if not isinstance(config_dict, dict):
            raise ValueError("IndexTTS2 config.yaml must contain a top-level mapping.")

        config_dict.setdefault("model_type", "indextts2")
        config_dict.setdefault("architectures", ["IndexTTS2Model"])

        config = IndexTTS2Config()
        for key, value in config_dict.items():
            setattr(config, key, value)

        return config_dict, config
