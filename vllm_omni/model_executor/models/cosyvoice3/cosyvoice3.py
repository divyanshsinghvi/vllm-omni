import os
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
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
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
from vllm_omni.model_executor.models.cosyvoice3.utils import (
    concat_text_with_prompt_ids,
    extract_speech_feat,
    extract_speech_token,
    extract_spk_embedding,
    extract_text_token,
    make_pad_mask,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class CosyVoice3MultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        """If the config is not already present pass it
        as a class and it will try to find it in your
        model directory just copy the config class there also.
        """
        return self.ctx.get_hf_config(CosyVoice3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """How many audio can you pass. I think I should keep it as 1
        For now I have kept it None.
        """
        return {"audio": None}


class CosyVoice3MultiModalProcessor(BaseMultiModalProcessor[CosyVoice3MultiModalProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        apply-> cached_apply_hf_processor -> apply_hf_processor_mm ->
        _call_hf_processor.
        _call_hf_processor takes input prompt and mm_data and returns
        token ids and tensors
        """
        import onnxruntime

        from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer

        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        self.tokenizer = get_qwen_tokenizer(
            token_path=os.path.join(model_dir, config.qwen_pretrain_path),
            skip_special_tokens=config.skip_special_tokens,
            version=config.version,
        )

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.speech_tokenizer = onnxruntime.InferenceSession(
            os.path.join(model_dir, config.speech_tokenizer_path),
            sess_options=option,
            providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"],
        )

        from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram

        feat_cfg = getattr(config, "feat_extractor", {})
        self.feat_extractor = partial(mel_spectrogram, **feat_cfg)
        campplus_full_path = os.path.join(model_dir, config.campplus_onxx_path)
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_full_path, sess_options=option, providers=["CPUExecutionProvider"]
        )

        audio = mm_data.get("audio", None)

        if audio is None:
            audio = mm_data.get("audios")
            if audio is not None:
                audio = audio[0], config.target_sr

        text_token, text_token_len = extract_text_token(prompt, self.tokenizer, config.allowed_special)
        if audio is None:
            # Text-only path for profiling/cache
            return BatchFeature({"input_ids": text_token, "input_len": [text_token_len]})

        prompt_text = mm_kwargs.get("prompt_text")

        if not isinstance(prompt_text, str):
            raise ValueError(f"prompt text is None : {prompt_text}")

        prompt_text_token, prompt_text_token_len = extract_text_token(
            prompt_text, self.tokenizer, config.allowed_special
        )

        input_ids, input_len = concat_text_with_prompt_ids(
            text_token,
            text_token_len,
            prompt_text_token,
            prompt_text_token_len,
        )
        logger.debug(
            "cosyvoice _call_hf_processor: prompt_text_token=%s text_token=%s input_ids=%s "
            "prompt_text_len=%s text_len=%s input_len=%s",
            prompt_text_token.tolist(),
            text_token.tolist(),
            input_ids.tolist(),
            int(prompt_text_token_len),
            int(text_token_len),
            int(input_len),
        )
        device = "cpu"

        speech_token, speech_token_len = extract_speech_token(audio, self.speech_tokenizer, device)
        speech_feat, speech_feat_len = extract_speech_feat(audio, self.feat_extractor, device)

        if config.sample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, : 2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len

        embedding = extract_spk_embedding(audio, self.campplus_session, device)

        ft = BatchFeature(
            {
                "input_ids": input_ids,
                "input_len": [input_len],
                "text_len": [text_token_len],
                "prompt_text_token": prompt_text_token,
                "prompt_text_len": [prompt_text_token_len],
                "speech_feat": speech_feat,
                "speech_feat_len": [speech_feat_len],
                "speech_token": speech_token,
                "speech_token_len": [speech_token_len],
                "embedding": embedding,
            }
        )

        return ft

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_len": MultiModalFieldConfig.batched("audio"),
            "text_len": MultiModalFieldConfig.batched("audio"),
            "prompt_text_len": MultiModalFieldConfig.batched("audio"),
            "prompt_text_token": MultiModalFieldConfig.batched("audio"),
            "speech_feat": MultiModalFieldConfig.batched("audio"),
            "speech_feat_len": MultiModalFieldConfig.batched("audio"),
            "speech_token": MultiModalFieldConfig.batched("audio"),
            "speech_token_len": MultiModalFieldConfig.batched("audio"),
            "embedding": MultiModalFieldConfig.batched("audio"),
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
            # TODO: Think if this can be done better
            # sos + task + audio token ... ideally this needs to be split into
            # two start and end but somehow I couldn't pass two of these
            # wutg target .start() and .end()
            token_len = out_mm_kwargs["audio"][0]["speech_token_len"].data[0].item()
            return [1] * (1 + 1 + token_len)

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
        return MultiModalDataParser(target_sr=self.info.ctx.get_hf_config().target_sr)


class CosyVoice3DummyInputsBuilder(BaseDummyInputsBuilder[CosyVoice3MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a test of the CosyVoice3 system capability."

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio")
        max_prompt_seconds = 30
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
                24000,
            ),
        }
        return mm_data

    def get_dummy_processor_inputs(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        inputs.hf_processor_mm_kwargs = {"prompt_text": "Testing my voices. Why should I not?"}
        return inputs


@MULTIMODAL_REGISTRY.register_processor(
    CosyVoice3MultiModalProcessor,
    info=CosyVoice3MultiModalProcessingInfo,
    dummy_inputs=CosyVoice3DummyInputsBuilder,
)
class CosyVoice3Model(
    nn.Module,
    SupportsMultiModal,
):
    supports_multimodal_raw_input_only = True
    supports_multimodal = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.have_multimodal_outputs = True
        self.model_stage = vllm_config.model_config.model_stage
        self.model_dir = vllm_config.model_config.model
        self.model = None
        if self.model_stage == "talker":
            # Initialize talker stage (text to speech tokens)

            from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_talker import CosyVoice3LM, Qwen2Encoder

            llm = Qwen2Encoder(os.path.join(self.model_dir, self.config.llm["llm"]["pretrain_path"]))
            self.text_speech_lm_model = CosyVoice3LM(
                llm_input_size=self.config.llm["llm_input_size"],
                llm_output_size=self.config.llm["llm_output_size"],
                speech_token_size=self.config.llm["speech_token_size"],
                llm=llm,
                length_normalized_loss=self.config.llm["length_normalized_loss"],
                lsm_weight=self.config.llm["lsm_weight"],
                mix_ratio=self.config.llm["mix_ratio"],
            )
            self.llm_cache = None
            self.model = self.text_speech_lm_model
        elif self.model_stage == "code2wav":
            # Initialize code2wav stage (flow matching + vocoder)
            from omegaconf import DictConfig

            from vllm_omni.model_executor.models.cosyvoice3.dit import DiT
            from vllm_omni.model_executor.models.cosyvoice3.flow import (
                CausalConditionalCFM,
                CausalMaskedDiffWithDiT,
                PreLookaheadLayer,
            )

            # Initialize acoustic features to waveform stage
            from vllm_omni.model_executor.models.cosyvoice3.hifigan import CausalConvRNNF0Predictor, CausalHiFTGenerator

            pre_lookahead_layer = PreLookaheadLayer(**self.config.flow["pre_lookahead_layer"])

            decoder_cfg = self.config.flow["decoder"]
            cfm_params = DictConfig(decoder_cfg["cfm_params"])
            estimator = DiT(**decoder_cfg["estimator"])
            decoder = CausalConditionalCFM(
                in_channels=decoder_cfg["in_channels"],
                estimator=estimator,
                cfm_params=cfm_params,
                n_spks=decoder_cfg["n_spks"],
                spk_emb_dim=decoder_cfg["spk_emb_dim"],
            )
            self.chunk_aware_flow_matching_model = CausalMaskedDiffWithDiT(
                input_size=self.config.flow["input_size"],
                output_size=self.config.flow["output_size"],
                spk_embed_dim=self.config.flow["spk_embed_dim"],
                output_type=self.config.flow["output_type"],
                vocab_size=self.config.flow["vocab_size"],
                input_frame_rate=self.config.flow["input_frame_rate"],
                only_mask_loss=self.config.flow["only_mask_loss"],
                token_mel_ratio=self.config.flow["token_mel_ratio"],
                pre_lookahead_len=self.config.flow["pre_lookahead_len"],
                pre_lookahead_layer=pre_lookahead_layer,
                decoder=decoder,
            )
            self.model = self.chunk_aware_flow_matching_model

            f0_predictor = CausalConvRNNF0Predictor(
                num_class=self.config.hift["f0_predictor"]["num_class"],
                in_channels=self.config.hift["f0_predictor"]["in_channels"],
                cond_channels=self.config.hift["f0_predictor"]["cond_channels"],
            )
            self.hift = CausalHiFTGenerator(
                in_channels=self.config.hift["in_channels"],
                base_channels=self.config.hift["base_channels"],
                nb_harmonics=self.config.hift["nb_harmonics"],
                sampling_rate=self.config.hift["sampling_rate"],
                nsf_alpha=self.config.hift["nsf_alpha"],
                nsf_sigma=self.config.hift["nsf_sigma"],
                nsf_voiced_threshold=self.config.hift["nsf_voiced_threshold"],
                upsample_rates=self.config.hift["upsample_rates"],
                upsample_kernel_sizes=self.config.hift["upsample_kernel_sizes"],
                istft_params=self.config.hift["istft_params"],
                resblock_kernel_sizes=self.config.hift["resblock_kernel_sizes"],
                resblock_dilation_sizes=self.config.hift["resblock_dilation_sizes"],
                source_resblock_kernel_sizes=self.config.hift["source_resblock_kernel_sizes"],
                source_resblock_dilation_sizes=self.config.hift["source_resblock_dilation_sizes"],
                lrelu_slope=self.config.hift["lrelu_slope"],
                audio_limit=self.config.hift["audio_limit"],
                conv_pre_look_right=self.config.hift["conv_pre_look_right"],
                f0_predictor=f0_predictor,
            )
            # Run hift in float32 to avoid dtype mismatches in internal ops.
            self.hift = self.hift.float()
            self.token_overlap_len = 20
            self.mel_overlap_len = int(
                self.token_overlap_len / self.chunk_aware_flow_matching_model.input_frame_rate * 22050 / 256
            )
            self.mel_window = np.hamming(2 * self.mel_overlap_len)
            self.mel_cache_len = 20
            self.source_cache_len = int(self.mel_cache_len * 256)
            self.speech_window = np.hamming(2 * self.source_cache_len)
        else:
            raise ValueError(f"Model stage not supported {self.model_stage}")

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == "talker":
            logits = self.model.llm_decoder(hidden_states)
            vocab_size = self.config.vocab_size
            pad_size = vocab_size - logits.size(-1)
            pad_shape = logits.shape[:-1] + (pad_size,)
            pad = logits.new_full(pad_shape, float("-inf"))
            eos_token_val = logits[..., self.config.llm["eos_token_id"]].clone()
            logits[..., -200:] = float("-inf")
            logits[..., self.config.llm["eos_token_id"]] = eos_token_val
            logits = torch.cat([logits, pad], dim=-1)
            return logits
        else:
            raise RuntimeError(f"embed_input_ids is only valid for {self.model_stage}.")

    def embed_multimodal(self, **kwargs: object) -> torch.Tensor:
        if self.model_stage == "talker":
            self.speech_token = kwargs["speech_token"]
            self.embedding = kwargs["embedding"]
            self.speech_feat = kwargs["speech_feat"]
            return self.speech_token
        else:
            raise RuntimeError(f"embed_input_ids is only valid for {self.model_stage}.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "talker":
            if is_multimodal is not None and any(is_multimodal):
                embed_tokens = self.model.llm.model.model.embed_tokens(input_ids)
                sos = self.model.speech_embedding.weight[self.model.sos].reshape(1, -1)
                task_id = self.model.speech_embedding.weight[self.model.task_id].reshape(1, -1)
                pstoken = multimodal_embeddings[0][0]
                pstoken_len = len(pstoken)
                prompt_speech_token_emb = self.model.speech_embedding(pstoken)
                embed_tokens = torch.cat(
                    [sos, embed_tokens[2 + pstoken_len :], task_id, prompt_speech_token_emb], dim=0
                )
            else:
                embed_tokens = self.model.speech_embedding.weight[input_ids]
            return embed_tokens
        elif self.model_stage == "code2wav":
            assert input_ids.dim() == 1
            hidden = int(self.config.hidden_size)
            return torch.zeros(
                (input_ids.shape[0], hidden),
            )
        else:
            raise RuntimeError(f"embed_input_ids is not valid for {self.model_stage}.")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        if self.model_stage == "talker":
            if inputs_embeds is None and input_ids is not None:
                raise Exception(f"inputs_embeds {input_ids} {inputs_embeds}")

            # Ensure [B, T, C]
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            batch, seq_len, _ = inputs_embeds.shape
            if seq_len > 1:
                self.llm_cache = None

            if self.llm_cache is not None:
                seq_len += self.llm_cache[0][0].shape[2]
            masks = torch.tril(torch.ones((1, seq_len, seq_len), device=inputs_embeds.device)).to(torch.bool)
            hidden_states, self.llm_cache = self.model.llm.forward_one_step(inputs_embeds, masks, cache=self.llm_cache)
            ## TODO Shift to vllm attention backed
            hidden_states = hidden_states.squeeze(0)

            multimodal_outputs = {}

            if hasattr(self, "speech_token"):
                multimodal_outputs = {
                    "speech_token": self.speech_token,
                    "embedding": self.embedding,
                    "speech_feat": self.speech_feat,
                }

            return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs=multimodal_outputs)
        elif self.model_stage == "code2wav":
            runtime_info = kwargs.get("runtime_additional_information", [])

            if not runtime_info:
                length = 30 * 24000
                audio = np.zeros((length,))
                return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audio})

            d = next(self.parameters())
            device, dtype = d.device, d.dtype
            embedding = runtime_info[0]["embedding"][0].to(device=device, dtype=dtype)
            embedding = F.normalize(embedding, dim=1)
            embedding = self.model.spk_embed_affine_layer(embedding)

            prompt_token = runtime_info[0]["speech_token"][0].to(device=device)
            # This is done to remove the last eos token.
            input_ids = input_ids[..., :-1]

            token = input_ids.unsqueeze(0).to(device=device)
            token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
            # Build length tensors for pad mask logic.
            prompt_token_len = torch.tensor([token_len1], device=token.device, dtype=torch.int32)
            token_len = torch.tensor([token_len2], device=token.device, dtype=torch.int32)
            token = torch.concat([prompt_token, token], dim=1)
            token_len = prompt_token_len + token_len
            mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
            token = self.model.input_embedding(torch.clamp(token, min=0)) * mask
            # text encode
            prompt_feat = runtime_info[0]["speech_feat"][0]

            h = self.model.pre_lookahead_layer(token)
            h = h.repeat_interleave(self.model.token_mel_ratio, dim=1)
            mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

            # get conditions
            conds = torch.zeros([1, mel_len1 + mel_len2, self.model.output_size], device=token.device).to(h.dtype)

            conds[:, :mel_len1] = prompt_feat

            conds = conds.transpose(1, 2)

            mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
            feat, _ = self.model.decoder(
                mu=h.transpose(1, 2).contiguous(),
                mask=mask.unsqueeze(1),
                spks=embedding,
                cond=conds,
                n_timesteps=10,
                streaming=False,
            )

            feat = feat[:, :, mel_len1:]

            tts_mel = feat

            token_offset = 0
            tts_mel = tts_mel[:, :, token_offset * self.model.token_mel_ratio :]

            # TODO Add speed control later
            # if speed != 1.0:
            #     tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear")
            hift_weight = self.hift.m_source.l_linear.weight
            tts_mel = tts_mel.to(device=hift_weight.device, dtype=hift_weight.dtype)
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel)

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": tts_speech},
            )
        else:
            raise ValueError(f"Stop it! {input_ids}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_stage == "talker":
            # Load weights for text to speech LM stage
            llm_weight_path = os.path.join(self.model_dir, "llm.pt")
            device = next(self.parameters()).device
            self.model.load_state_dict(torch.load(llm_weight_path, map_location=device), strict=True)
            self.model.to(device).eval()
        elif self.model_stage == "code2wav":
            # Load weights for chunk aware flow matching stage
            flow_weight_path = os.path.join(self.model_dir, "flow.pt")
            device = next(self.parameters()).device
            self.model.load_state_dict(torch.load(flow_weight_path, map_location=device), strict=True)
            self.model.to(device).eval()

            hift_weight_path = os.path.join(self.model_dir, "hift.pt")
            hift_state_dict = {
                k.replace("generator.", ""): v for k, v in torch.load(hift_weight_path, map_location=device).items()
            }
            self.hift.load_state_dict(hift_state_dict, strict=True)
            self.hift.to(device).eval()
        else:
            raise ValueError(f"{self.model_stage} not supported yet!")
