import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torchaudio
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
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


class IndexTTS2ProcessingInfo(BaseProcessingInfo):
    """Processing info for IndexTTS2 model - text processing only."""

    def get_hf_config(self):
        return self.ctx.get_hf_config(IndexTTS2Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}  # Use "audio" modality for text_token_ids field


class IndexTTS2Processor(BaseMultiModalProcessor[IndexTTS2ProcessingInfo]):
    """Processor for IndexTTS2 - handles text tokenization only."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Use the custom tokenizer
        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        bpe_path = os.path.join(model_dir, config.dataset["bpe_model"])

        from indextts.utils.front import TextNormalizer, TextTokenizer

        normalizer = TextNormalizer(enable_glossary=True)
        normalizer.load()
        tokenizer = TextTokenizer(bpe_path, normalizer=normalizer)

        text_tokens = tokenizer.tokenize(prompt)
        text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        input_ids = torch.tensor([text_token_ids], dtype=torch.long)  # 2D with batch dim

        return BatchFeature(
            {
                "input_ids": input_ids,
                "text_token_ids": input_ids,  # Pass through kwargs for forward()
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
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
        return []  # No prompt modifications

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=22050)


class IndexTTS2DummyInputsBuilder(BaseDummyInputsBuilder[IndexTTS2ProcessingInfo]):
    """Dummy inputs builder for profiling IndexTTS2."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a test."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> dict:
        return {}  # No multimodal data needed for profiling

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        return ProcessorInputs(
            prompt=self.get_dummy_text(mm_counts),
            mm_data={},
            hf_processor_mm_kwargs={},
        )


@MULTIMODAL_REGISTRY.register_processor(
    IndexTTS2Processor,
    info=IndexTTS2ProcessingInfo,
    dummy_inputs=IndexTTS2DummyInputsBuilder,
)
class IndexTTS2Model(nn.Module):
    """IndexTTS2 TTS model with GPT and S2Mel stages."""

    supports_multimodal = True
    have_multimodal_outputs = True
    requires_raw_input_tokens = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.cfg = vllm_config.model_config.hf_config
        self.model_dir = vllm_config.model_config.model
        self.model_stage = vllm_config.model_config.model_stage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_stage == "gpt":
            from indextts.gpt.model_v2 import UnifiedVoice

            self.talker = UnifiedVoice(**self.cfg.gpt)
            self.stop_mel_token = self.cfg.gpt["stop_mel_token"]

            self.semantic_model = None
            self.feature_extractor = None
            self.semantic_mean = None
            self.semantic_std = None
            self.semantic_codec = None
            self.campplus_model = None
            self.mel_fn = None
        elif self.model_stage == "s2mel":
            from types import SimpleNamespace

            from indextts.s2mel.modules.commons import MyModel

            def dict_to_namespace(d):
                """Recursively convert dict to SimpleNamespace for attribute access."""
                if isinstance(d, dict):
                    return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
                return d

            self.s2mel = MyModel(dict_to_namespace(self.cfg.s2mel), use_gpt_latent=True)
            self.semantic_codec = None
            self.bigvgan = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        if self.model_stage == "gpt":
            runtime_info = kwargs.get("runtime_additional_information", [{}])
            if isinstance(runtime_info, list) and len(runtime_info) > 0:
                runtime_info = runtime_info[0]

            audio_path = runtime_info.get("audio_path", [None])
            if isinstance(audio_path, list):
                audio_path = audio_path[0] if audio_path else None

            emo_audio_path = runtime_info.get("emo_audio_path", [None])
            if isinstance(emo_audio_path, list):
                emo_audio_path = emo_audio_path[0] if emo_audio_path else None

            emo_weight = runtime_info.get("emo_weight", [0.5])
            if isinstance(emo_weight, list):
                emo_weight = emo_weight[0] if emo_weight else 0.5

            model_param = next(self.talker.parameters())
            target_device = model_param.device
            target_dtype = model_param.dtype

            if audio_path and os.path.exists(audio_path):
                audio, sr = torchaudio.load(audio_path)
                audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio).to(target_device)
                audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio).to(target_device)

                spk_cond_emb = self.extract_speaker_embedding(audio, int(sr))
                _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
                ref_mel = self.mel_fn(audio_22k.float())

                feat = torchaudio.compliance.kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
                feat = feat - feat.mean(dim=0, keepdim=True)
                style = self.campplus_model(feat.unsqueeze(0))
            else:
                # Dummy for profile run
                spk_cond_emb = torch.zeros(
                    1, 100, self.cfg.semantic_codec["hidden_size"], device=target_device, dtype=target_dtype
                )
                S_ref = torch.zeros(1, 8, 100, device=target_device, dtype=torch.long)
                ref_mel = torch.zeros(1, 80, 100, device=target_device, dtype=target_dtype)
                style = torch.zeros(1, 192, device=target_device, dtype=target_dtype)

            if emo_audio_path and os.path.exists(emo_audio_path):
                emo_audio, emo_sr = torchaudio.load(emo_audio_path)
                emo_cond_emb = self.extract_speaker_embedding(emo_audio, int(emo_sr))
            else:
                emo_cond_emb = spk_cond_emb.clone()

            emo_alpha = torch.tensor([[emo_weight]], device=target_device, dtype=target_dtype)
            spk_cond_emb = spk_cond_emb.to(device=target_device, dtype=target_dtype)
            emo_cond_emb = emo_cond_emb.to(device=target_device, dtype=target_dtype)

            B = spk_cond_emb.shape[0]
            cond_lengths = torch.full((B,), spk_cond_emb.shape[1], device=spk_cond_emb.device)
            emo_cond_lengths = torch.full((B,), emo_cond_emb.shape[1], device=emo_cond_emb.device)

            emovec = self.talker.merge_emovec(
                spk_cond_emb, emo_cond_emb, cond_lengths, emo_cond_lengths, alpha=emo_alpha
            )

            top_p = runtime_info.get("top_p", [0.8])
            top_p = top_p[0] if isinstance(top_p, list) else top_p
            top_k = runtime_info.get("top_k", [30])
            top_k = top_k[0] if isinstance(top_k, list) else top_k
            temperature = runtime_info.get("temperature", [0.8])
            temperature = temperature[0] if isinstance(temperature, list) else temperature
            autoregressive_batch_size = 1
            length_penalty = runtime_info.get("length_penalty", [0.0])
            length_penalty = length_penalty[0] if isinstance(length_penalty, list) else length_penalty
            num_beams = runtime_info.get("num_beams", [3])
            num_beams = num_beams[0] if isinstance(num_beams, list) else num_beams
            repetition_penalty = runtime_info.get("repetition_penalty", [10.0])
            repetition_penalty = repetition_penalty[0] if isinstance(repetition_penalty, list) else repetition_penalty
            max_mel_tokens = runtime_info.get("max_mel_tokens", [1500])
            max_mel_tokens = max_mel_tokens[0] if isinstance(max_mel_tokens, list) else max_mel_tokens

            text_token_ids = kwargs.get("text_token_ids")

            if text_token_ids is None:
                # Profile run - return dummy output
                model_dim = self.cfg.gpt["model_dim"]
                return OmniOutput(
                    text_hidden_states=torch.zeros(1, 1, model_dim, device=target_device, dtype=target_dtype),
                    multimodal_outputs={
                        "latent": torch.zeros(1, 1, model_dim, device=target_device, dtype=target_dtype),
                        "speech_conditioning_latent": torch.zeros(
                            1, 1, model_dim, device=target_device, dtype=target_dtype
                        ),
                        "codes": torch.zeros(1, 1, device=target_device, dtype=torch.long),
                        "code_lens": torch.tensor([1], device=target_device),
                        "S_ref": torch.zeros(8, 1, 1, device=target_device, dtype=torch.long),  # [N, B, T]
                        "ref_mel": torch.zeros(1, 80, 1, device=target_device, dtype=target_dtype),
                        "style": torch.zeros(1, 192, device=target_device, dtype=target_dtype),
                    },
                )

            self.talker.inference_model.eval().to(dtype=target_dtype)

            codes, speech_conditioning_latent = self.talker.inference_speech(
                spk_cond_emb,
                text_token_ids,
                emo_cond_emb,
                cond_lengths=torch.tensor([spk_cond_emb.shape[1]], device=spk_cond_emb.device),
                emo_cond_lengths=torch.tensor([emo_cond_emb.shape[1]], device=spk_cond_emb.device),
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
            latent = self.talker(
                speech_conditioning_latent,
                text_token_ids,
                torch.tensor([text_token_ids.shape[-1]], device=target_device),
                codes,
                torch.tensor([codes.shape[-1]], device=target_device),
                emo_cond_emb,
                cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=target_device),
                emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[1]], device=target_device),
                emo_vec=emovec,
                use_speed=use_speed,
            )

            multimodal_outputs = {
                "latent": latent,
                "speech_conditioning_latent": speech_conditioning_latent,
                "codes": codes,
                "code_lens": code_lens,
                "S_ref": S_ref,
                "ref_mel": ref_mel,
                "style": style,
            }

            return OmniOutput(text_hidden_states=multimodal_outputs["latent"], multimodal_outputs=multimodal_outputs)

        elif self.model_stage == "s2mel":
            runtime_info = kwargs.get("runtime_additional_information", [])

            if not runtime_info:
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"audio": torch.zeros(1, 22050), "sample_rate": 22050},
                )

            info = runtime_info[0]
            latent = info["latent"]
            codes = info["codes"]
            code_lens = info["code_lens"]
            S_ref = info["S_ref"]
            ref_mel = info["ref_mel"]
            style = info["style"]

            model_param = next(self.s2mel.parameters())
            target_device = model_param.device
            target_dtype = model_param.dtype

            latent = latent.to(device=target_device, dtype=target_dtype)
            codes = codes.to(device=target_device).long()
            code_lens = code_lens.to(device=target_device).long()
            S_ref = S_ref.to(device=target_device).long()
            ref_mel = ref_mel.to(device=target_device, dtype=target_dtype)
            style = style.to(device=target_device, dtype=target_dtype)

            ref_target_lengths = torch.LongTensor([ref_mel.size(-1)]).to(target_device)
            S_ref_emb = self.semantic_codec.quantizer.vq2emb(S_ref.long(), n_quantizers=3)
            S_ref_emb = S_ref_emb.transpose(1, 2)
            prompt_condition = self.s2mel.models["length_regulator"](
                S_ref_emb, ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0]

            latent = self.s2mel.models["gpt_layer"](latent)

            S_infer = self.semantic_codec.quantizer.vq2emb(codes.long().unsqueeze(1))
            S_infer = S_infer.transpose(1, 2)
            S_infer = S_infer + latent

            target_lengths = (code_lens * 1.72).long()
            cond = self.s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]

            cat_condition = torch.cat([prompt_condition, cond], dim=1)

            diffusion_steps = 25
            inference_cfg_rate = 0.7
            vc_target = self.s2mel.models["cfm"].inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(target_device),
                ref_mel,
                style,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, ref_mel.size(-1) :]

            wav = self.bigvgan(vc_target.float()).squeeze(1)
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": wav.cpu(), "sample_rate": 22050},
            )
        else:
            raise Exception("Oops")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.model_stage == "gpt":
            return self.talker.text_embedding(input_ids)
        elif self.model_stage == "s2mel":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.cfg.hidden_size)
        raise NotImplementedError(f"embed_input_ids not implemented for stage {self.model_stage}")

    def embed_multimodal(self, **kwargs: Any) -> None:
        """Audio conditioning is passed via runtime_additional_information."""
        return None

    def extract_speaker_embedding(self, audio: torch.Tensor, sr: int = 22050) -> torch.Tensor:
        """Extract speaker embedding using Wav2Vec2BertModel."""
        if sr != 16000:
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        else:
            audio_16k = audio

        if audio_16k.dim() == 1:
            audio_np = audio_16k.cpu().numpy()
        else:
            audio_np = audio_16k.squeeze().cpu().numpy()

        inputs = self.feature_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
        model_dtype = next(self.semantic_model.parameters()).dtype
        input_features = inputs["input_features"].to(device=self.device, dtype=model_dtype)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.inference_mode():
            outputs = self.semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = outputs.hidden_states[17]
            feat = (feat - self.semantic_mean) / self.semantic_std

        return feat

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_stage == "gpt":
            from indextts.utils.checkpoint import load_checkpoint

            self.talker_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
            load_checkpoint(self.talker, self.talker_path)
            self.talker.to(self.device)
            self.talker.eval().half()
            self.talker.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=True)

            from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

            self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
            self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", torch_dtype=torch.float16)
            self.semantic_model.to(self.device).eval()

            stat_path = os.path.join(self.model_dir, self.cfg.w2v_stat)
            stat_mean_var = torch.load(stat_path, map_location=self.device)
            self.semantic_mean = stat_mean_var["mean"].to(self.device)
            self.semantic_std = torch.sqrt(stat_mean_var["var"]).to(self.device)

            import safetensors.torch
            from huggingface_hub import hf_hub_download
            from indextts.utils.maskgct_utils import build_semantic_codec

            semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
            semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
            safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
            self.semantic_codec = semantic_codec.to(self.device).eval()

            from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

            campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
            self.campplus_model = self.campplus_model.to(self.device).eval()

            from indextts.s2mel.modules.audio import mel_spectrogram

            mel_fn_args = {
                "n_fft": self.cfg.s2mel["preprocess_params"]["spect_params"]["n_fft"],
                "win_size": self.cfg.s2mel["preprocess_params"]["spect_params"]["win_length"],
                "hop_size": self.cfg.s2mel["preprocess_params"]["spect_params"]["hop_length"],
                "num_mels": self.cfg.s2mel["preprocess_params"]["spect_params"]["n_mels"],
                "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
                "fmin": self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmin", 0),
                "fmax": None
                if self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
                else 8000,
                "center": False,
            }
            self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

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

            import safetensors.torch
            from huggingface_hub import hf_hub_download
            from indextts.utils.maskgct_utils import build_semantic_codec

            semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
            semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
            safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
            self.semantic_codec = semantic_codec.to(self.device).eval()

            from indextts.s2mel.modules.bigvgan import bigvgan as bigvgan_module

            bigvgan_name = self.cfg.vocoder["name"]
            self.bigvgan = bigvgan_module.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            self.bigvgan = self.bigvgan.to(self.device)
            self.bigvgan.remove_weight_norm()
            self.bigvgan.eval()

        else:
            raise ValueError(f"Unknown model_stage: {self.model_stage}")
