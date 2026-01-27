import os
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchaudio

# Suppress verbose logging from indextts/WeTextProcessing
# The tn library creates loggers with INFO level and adds handlers each time,
# so we use a filter that can't be overridden
# class _BlockAllFilter(logging.Filter):
#     def filter(self, record):
#         return False
# # Pre-configure wetext loggers BEFORE any tn imports
# for _name in ["wetext-zh_normalizer", "wetext-en_normalizer"]:
#     _logger = logging.getLogger(_name)
#     _logger.addFilter(_BlockAllFilter())
#     _logger.propagate = False  # Prevent propagation to root logger
# _original_level = logging.root.level
# logging.root.setLevel(logging.WARNING)
from indextts.utils.front import TextNormalizer, TextTokenizer

# logging.root.setLevel(_original_level)
# # Keep these loggers quiet for subsequent calls
# logging.getLogger("WETEXT").setLevel(logging.WARNING)
# logging.getLogger("WeTextProcessing").setLevel(logging.WARNING)
# logging.getLogger("indextts").setLevel(logging.WARNING)
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs

# from vllm.model_executor.models.qwen2 import
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import TokenizerRegistry

from vllm_omni.model_executor.models.indextts2.index_tts_config import IndexTTS2Config
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

TokenizerRegistry.register(
    "indextts2",
    module="vllm_omni.model_executor.models.indextts2.tokenizer",
    class_name="IndexTTS2Tokenizer",
)


class IndexTTS2MultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(IndexTTS2Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class IndexTTS2MultiModalProcessor(BaseMultiModalProcessor[IndexTTS2MultiModalProcessingInfo]):
    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        feat = vq_emb.hidden_states[17]  # (B, T, C)

        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        logger.info(f"prompt: {prompt} mm_data: {mm_data} mm_kwargs: {mm_kwargs} tok_kwargs: {tok_kwargs}")
        self.config = self.info.ctx.get_hf_config()
        self.model_dir = self.info.ctx.model_config.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(self.model_dir)

        self.bpe_path = os.path.join(self.model_dir, self.config.dataset["bpe_model"])
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        logger.info(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        logger.info(f">> bpe model loaded from: {self.bpe_path}")

        # tokenizer_path = config.tokenizer_path
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        text_tokens_list = self.tokenizer.tokenize(prompt)
        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        input_ids = torch.tensor([text_token_ids], dtype=torch.long, device=device)

        logger.info(f"input_ids {input_ids}")

        if len(mm_data) == 0:
            return BatchFeature({"input_ids": input_ids})

        logger.info(f"{mm_kwargs}")
        # Currently only one mode for index_tts is supported
        # emo_mode = mm_kwargs.get("emo_mode")
        emo_audio, _ = mm_kwargs.get("emo_audio")
        emo_weight = mm_kwargs.get("emo_weight")

        # print(mm_data.get("audios")[0])
        audio = mm_data.get("audios")[0]
        sr = 22050
        # print(type(audio))
        # print(device)

        audio = torch.Tensor(audio)
        emo_audio = torch.Tensor(emo_audio)
        # print("Audio")
        # print(emo_audio.mean())
        # audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        # print(audio_16k.mean())
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        # logger.info(f"input_features {input_features}")

        with torch.inference_mode():
            self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
            self.semantic_model.eval()
            self.semantic_model.to(device)
            self.semantic_model.eval()
            self.stat_path = os.path.join(self.model_dir, self.config.w2v_stat)
            stat_mean_var = torch.load(self.stat_path)
            self.semantic_mean = stat_mean_var["mean"].to(device)
            self.semantic_std = torch.sqrt(stat_mean_var["var"]).to(device)

            spk_cond_emb = self.get_emb(input_features, attention_mask)

            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"].to(device)
            emo_attention_mask = emo_inputs["attention_mask"].to(device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            logger.info(f"emo_cond_emb {input_ids.shape}")
            # emo_cond_emb = emo_cond_emb.squeeze()
            # spk_cond_emb = spk_cond_emb.squeeze()

            # logger.info(f"emo_cond_emb {input_ids.shape}")
            # logger.info(f"emo_cond_emb {emo_cond_emb.shape}")

        return BatchFeature(
            {
                "input_ids": input_ids.cpu(),
                "text_token_ids": input_ids.cpu(),  # extra field
                "spk_cond_emb": spk_cond_emb.cpu(),
                "emo_cond_emb": emo_cond_emb.cpu(),
                "emo_weight": torch.tensor([emo_weight]),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        if "spk_cond_emb" in hf_inputs:
            logger.info(
                f"hf_inputs {hf_inputs['spk_cond_emb'].shape} "
                f"{hf_inputs['emo_cond_emb'].shape} {hf_inputs['emo_weight']}"
            )
            logger.info(f"hf_processor_mm_kwargs {hf_processor_mm_kwargs}")
        return {
            "spk_cond_emb": MultiModalFieldConfig.batched("audio"),
            "emo_cond_emb": MultiModalFieldConfig.batched("audio"),
            "emo_weight": MultiModalFieldConfig.batched("audio"),
            "text_token_ids": MultiModalFieldConfig.batched("audio"),
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def insertion_end(item_idx):
            return [1]

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.start(),
                insertion=insertion_end,
            ),
        ]

    def _get_data_parser(self) -> MultiModalDataParser:
        """For audio you need to define target_sr;
        so need to create this data parser to avoid those errors
        """
        return MultiModalDataParser(target_sr=self.info.ctx.get_hf_config().s2mel["preprocess_params"]["sr"])


class IndexTTS2DummyInputsBuilder(BaseDummyInputsBuilder[IndexTTS2MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a test of the IndexTTS2 system capability."

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio")
        max_prompt_seconds = 15
        prompt_sample_rate = 24000
        target_audio_length = max_prompt_seconds * prompt_sample_rate

        audio_overrides = mm_options.get("audio") if mm_options else None
        mm_data = {
            "audio": (
                self._get_dummy_audios(
                    length=target_audio_length,
                    num_audios=num_audios,
                    overrides=audio_overrides,
                )[0],
                prompt_sample_rate,
            ),
        }
        # TODO: Return this instead of mm_data
        print(mm_data)
        import librosa

        audio_signal, sr = librosa.load("/home/divyansh/code/open_source/vllm-omni/prompt.wav")
        audio_data = (audio_signal.astype(np.float32), sr)
        mm_data = {"audio": audio_data}
        return mm_data

    def get_dummy_processor_inputs(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        # num_audios = 1
        # max_prompt_seconds = 15
        # prompt_sample_rate = 16000
        # target_audio_length = max_prompt_seconds * prompt_sample_rate
        import librosa

        audio_signal, sr = librosa.load("/home/divyansh/code/open_source/vllm-omni/prompt.wav", sr=16000)
        audio_data = (audio_signal.astype(np.float32), sr)

        inputs.hf_processor_mm_kwargs = {
            "emo_mode": 1,
            "emo_weight": 1.0,
            # "emo_audio": (
            #     self._get_dummy_audios(
            #         length=target_audio_length,
            #         num_audios=num_audios,
            #     )[0],
            #     prompt_sample_rate,
            # ),
            "emo_audio": audio_data,
            "emo_text": "Abra Ka Dabra!",
        }
        return inputs


@MULTIMODAL_REGISTRY.register_processor(
    IndexTTS2MultiModalProcessor,
    info=IndexTTS2MultiModalProcessingInfo,
    dummy_inputs=IndexTTS2DummyInputsBuilder,
)
class IndexTTS2Model(nn.Module, SupportsMultiModal):
    supports_multimodal = True
    supports_multimodal_raw_input_only = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.cfg = vllm_config.model_config.hf_config
        self.model_dir = vllm_config.model_config.model
        self.model_stage = vllm_config.model_config.model_stage
        # self.cfg_dtype = vllm_config.model_config.dtype
        # self.use_accel = getattr(self.cfg, "use_accel", False)
        # TODO: Fix it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_stage == "gpt":
            from indextts.gpt.model_v2 import UnifiedVoice

            self.talker = UnifiedVoice(**self.cfg.gpt)

            logger.info("Unified Voice")
            self.stop_mel_token = self.cfg.gpt["stop_mel_token"]
        elif self.model_stage == "s2mel":
            from types import SimpleNamespace

            from indextts.s2mel.modules.commons import MyModel

            def dict_to_namespace(d):
                """Recursively convert dict to SimpleNamespace for attribute access."""
                if isinstance(d, dict):
                    return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
                return d

            self.s2mel = MyModel(dict_to_namespace(self.cfg.s2mel), use_gpt_latent=True)

            pass

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        if self.model_stage == "gpt":
            logger.info(
                f"input_ids {input_ids} positions {positions}"
                f"intermediate_tensors {intermediate_tensors}"
                f"inputs_embeds {inputs_embeds} "
                f"additional_information {additional_information}"
                f"kwargs {kwargs['emo_weight'].shape}"
                f"kwargs {kwargs['spk_cond_emb'].shape}"
                f"kwargs {kwargs['emo_cond_emb'].shape}"
                # f"kwargs {kwargs}"
            )

            # return inputs_embeds
            spk_cond_emb = kwargs.get("spk_cond_emb")
            emo_cond_emb = kwargs.get("emo_cond_emb")
            emo_alpha = kwargs.get("emo_weight")
            assert spk_cond_emb.dim() == 4
            assert emo_cond_emb.dim() == 4
            spk_cond_emb = spk_cond_emb.squeeze(1)
            emo_cond_emb = emo_cond_emb.squeeze(1)

            model_param = next(self.talker.parameters())
            target_device = model_param.device
            target_dtype = model_param.dtype
            print("Hello")
            print(target_dtype)
            print(target_device)

            spk_cond_emb = spk_cond_emb.to(device=target_device, dtype=target_dtype)
            emo_cond_emb = emo_cond_emb.to(device=target_device, dtype=target_dtype)
            emo_alpha = emo_alpha.to(device=target_device, dtype=target_dtype)

            B = spk_cond_emb.shape[0]
            cond_lengths = torch.full((B,), spk_cond_emb.shape[-1], device=spk_cond_emb.device)
            emo_cond_lengths = torch.full((B,), emo_cond_emb.shape[-1], device=emo_cond_emb.device)

            emovec = self.talker.merge_emovec(
                spk_cond_emb, emo_cond_emb, cond_lengths, emo_cond_lengths, alpha=emo_alpha
            )

            logger.info(f"emovec {emovec} {spk_cond_emb} {emo_cond_emb} ")

            ##TODO: Fix this value fetches
            top_p = kwargs.pop("top_p", 0.8)
            top_k = kwargs.pop("top_k", 30)
            temperature = kwargs.pop("temperature", 0.8)
            autoregressive_batch_size = 1
            length_penalty = kwargs.pop("length_penalty", 0.0)
            num_beams = kwargs.pop("num_beams", 3)
            repetition_penalty = kwargs.pop("repetition_penalty", 10.0)
            max_mel_tokens = kwargs.pop("max_mel_tokens", 1500)

            text_token_ids = kwargs.get("text_token_ids")
            text_token_ids = text_token_ids.squeeze(0)
            self.talker.inference_model.eval().to(dtype=target_dtype)

            logger.info(
                f"spk_cond_emb {spk_cond_emb.shape}\n"
                f"text_token_ids {text_token_ids.shape}\n"
                f"emo_cond_emb {emo_cond_emb.shape}\n"
                f"cond_lengths {cond_lengths.shape}\n"
                f"emo_cond_lengths {emo_cond_lengths.shape}\n"
                f"emo_vec {emovec.shape}\n"
                f"do_sample: True"
                f"top_p {top_p}"
                f"top_k {top_k}"
                f"temperature {temperature}"
                f"num_return_sequences {autoregressive_batch_size}"
                f"length_penalty {length_penalty}"
                f"num_beams {num_beams}"
                f"repetition_penalty {repetition_penalty}"
                f"max_generate_length {max_mel_tokens}"
            )

            # TODO: Fix caching in inference_model
            codes, speech_conditioning_latent = self.talker.inference_speech(
                spk_cond_emb,
                ### TODO: Fix it
                text_token_ids,
                emo_cond_emb,
                cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=spk_cond_emb.device),
                emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=spk_cond_emb.device),
                emo_vec=emovec,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=autoregressive_batch_size,
                length_penalty=length_penalty,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_generate_length=max_mel_tokens,
            )

            # code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
            code_lens = []
            max_code_len = 0
            for code in codes:
                if self.stop_mel_token not in code:
                    code_len = len(code)
                else:
                    len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                    code_len = len_[0].item() if len_.numel() > 0 else len(code)
                code_lens.append(code_len)
                max_code_len = max(max_code_len, code_len)
            codes = codes[:, :max_code_len]
            code_lens = torch.LongTensor(code_lens)
            code_lens = code_lens.to(self.device)

            use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
            # with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
            latent = self.talker(
                speech_conditioning_latent,
                text_token_ids,
                torch.tensor([text_token_ids.shape[-1]], device=text_token_ids.device),
                codes,
                torch.tensor([codes.shape[-1]], device=text_token_ids.device),
                emo_cond_emb,
                cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_token_ids.device),
                emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_token_ids.device),
                emo_vec=emovec,
                use_speed=use_speed,
            )

            multimodal_outputs = {
                "latent": latent,
                "speech_conditioning_latent": speech_conditioning_latent,
            }

            print(f"multimodal_outputs {multimodal_outputs}")

            return OmniOutput(text_hidden_states=multimodal_outputs["latent"], multimodal_outputs=multimodal_outputs)
        elif self.model_stage == "s2mel":
            # Get latent from runtime_additional_information (passed from gpt2s2mel stage input processor)
            runtime_info = kwargs.get("runtime_additional_information", [])
            logger.info(f"runtime_info {runtime_info}")
            if not runtime_info:
                length = 30 * 24000
                audio = np.zeros((length,))
                return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audio})

            info = runtime_info[0]
            latent = info.get("latent")
            speech_conditioning_latent = info.get("speech_conditioning_latent")

            logger.info(f"s2mel forward - latent shape: {latent.shape if latent is not None else None}")

            # Move to correct device/dtype
            model_param = next(self.s2mel.parameters())
            target_device = model_param.device
            target_dtype = model_param.dtype

            if isinstance(latent, torch.Tensor):
                latent = latent.to(device=target_device, dtype=target_dtype)
            if isinstance(speech_conditioning_latent, torch.Tensor):
                speech_conditioning_latent = speech_conditioning_latent.to(device=target_device, dtype=target_dtype)

            # Process through s2mel gpt_layer
            latent = self.s2mel.models["gpt_layer"](latent)

            # S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
            # S_infer = S_infer.transpose(1, 2)
            # S_infer = S_infer + latent
            # target_lengths = (code_lens * 1.72).long()

            # cond = self.s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
            # cat_condition = torch.cat([prompt_condition, cond], dim=1)
            # vc_target = self.s2mel.models["cfm"].inference(
            #     cat_condition,
            #     torch.LongTensor([cat_condition.size(1)]).to(cond.device),
            #     ref_mel,
            #     style,
            #     None,
            #     diffusion_steps,
            #     inference_cfg_rate=inference_cfg_rate,
            # )
            # vc_target = vc_target[:, :, ref_mel.size(-1) :]
            # s2mel_time += time.perf_counter() - m_start_time

            # m_start_time = time.perf_counter()
            # wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
            # print(wav.shape)
            # bigvgan_time += time.perf_counter() - m_start_time
            # wav = wav.squeeze(1)

            # # TODO: Add full s2mel inference (DiT, CFM, etc.)
            # multimodal_outputs = {
            #     "latent": latent,
            #     "speech_conditioning_latent": speech_conditioning_latent,
            # }

            # return OmniOutput(text_hidden_states=latent, multimodal_outputs=multimodal_outputs)
        else:
            raise Exception("Oops")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs into continuous representations.

        Required by vLLM's VllmModel interface.
        """
        if self.model_stage == "gpt":
            return self.talker.text_embedding(input_ids)
        raise NotImplementedError(f"embed_input_ids not implemented for stage {self.model_stage}")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Required by vLLM to recognize this as a generative model."""
        return None
        num_tokens = hidden_states.shape[0]
        vocab_size = self.cfg.gpt["number_mel_codes"]  # 8194
        stop_token_id = self.stop_mel_token  # 8193

        logits = torch.full(
            (num_tokens, vocab_size),
            fill_value=-1e9,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        logits[:, stop_token_id] = 0.0
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_stage == "gpt":
            from indextts.utils.checkpoint import load_checkpoint

            self.talker_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
            logger.info("loading gpt2")
            load_checkpoint(self.talker, self.talker_path)
            self.talker.to(self.device)
            # TODO : Fix fp16 / fp32
            self.talker.eval().half()
            # TODO : Fix fp16 / fp32
            logger.info("post init gpt2 config")
            self.talker.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=True)
        elif self.model_stage == "s2mel":
            from indextts.s2mel.modules.commons import load_checkpoint2

            s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
            self.s2mel, _, _, _ = load_checkpoint2(
                self.s2mel,
                None,
                s2mel_path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
            self.s2mel.to(self.device)
            self.s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            self.s2mel.eval()

        else:
            raise Exception("Oops")
