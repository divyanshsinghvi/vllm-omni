# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
AMD Micro-World Text-to-World (T2W) pipeline.

Generates action-controlled video from a text prompt and keyboard/mouse actions.
Based on the Wan2.1-T2V-1.3B architecture with ControlNet-style action injection.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.platforms import current_omni_platform

from .micro_world_transformer import MicroWorldControlNetTransformer

logger = logging.getLogger(__name__)


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=model_path, filename=f"{subfolder}/config.json")
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(config: dict) -> MicroWorldControlNetTransformer:
    """Create MicroWorldControlNetTransformer from a HF config dict."""
    kwargs: dict[str, Any] = {}

    # Map HF config keys to vllm-omni WanTransformer3DModel constructor args
    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_heads" in config:
        # Micro-World uses 'num_heads' and 'dim' rather than
        # 'num_attention_heads' and 'attention_head_dim'
        kwargs["num_attention_heads"] = config["num_heads"]
        if "dim" in config:
            kwargs["attention_head_dim"] = config["dim"] // config["num_heads"]
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_dim" in config:
        kwargs["in_channels"] = config["in_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_dim" in config:
        kwargs["out_channels"] = config["out_dim"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "freq_dim" in config:
        kwargs["freq_dim"] = config["freq_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "added_kv_proj_dim" in config:
        kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
    if "rope_max_seq_len" in config:
        kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]

    # Micro-World specific action params
    action_kwargs: dict[str, Any] = {}
    if "keyboard_dim" in config:
        action_kwargs["keyboard_dim"] = config["keyboard_dim"]
    if "mouse_dim" in config:
        action_kwargs["mouse_dim"] = config["mouse_dim"]
    if "action_dim" in config:
        action_kwargs["action_dim"] = config.get("action_dim", 1536)
    if "action_layers" in config and config["action_layers"] is not None:
        action_kwargs["action_layers"] = config["action_layers"]

    return MicroWorldControlNetTransformer(**action_kwargs, **kwargs)


def get_micro_world_t2w_post_process_func(od_config: OmniDiffusionConfig):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_micro_world_t2w_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process: validate action inputs from extra_params."""

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        # Actions come via extra_args; nothing to validate at this stage
        return request

    return pre_process_func


class MicroWorldT2WPipeline(nn.Module, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin):
    """AMD Micro-World Text-to-World pipeline.

    Generates action-controlled video from text prompts with keyboard/mouse inputs.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Weight sources
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Check for LoRA weights at model root
        lora_path = os.path.join(model, "lora_diffusion_pytorch_model.safetensors") if local_files_only else None
        if lora_path and os.path.exists(lora_path):
            logger.info(f"Found LoRA weights at {lora_path}")
            # LoRA will be handled by the diffusion LoRA manager

        # Text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)

        # VAE
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
        ).to(self.device)

        # Transformer
        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(transformer_config)

        # Store config for latent shape computation
        self.transformer_config = self.transformer.config

        # Scheduler
        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 3.0
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 8

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def load_weights(self, weights):
        """Load weights using AutoWeightsLoader for vLLM integration."""
        from vllm.model_executor.models.utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text prompts using UMT5 text encoder."""
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            neg_input_ids = neg_inputs.input_ids.to(device)
            neg_mask = neg_inputs.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(neg_input_ids, attention_mask=neg_mask)[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare noise latents for diffusion."""
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def _parse_actions(
        self,
        extra_args: dict[str, Any] | None,
        device: torch.device,
        dtype: torch.dtype,
        num_action_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract action tensors from extra_args.

        If actions are missing and ``num_action_frames`` is given, returns
        zero-filled defaults (used for warmup / dummy run, where actions
        are not meaningful).
        """
        mouse_actions = extra_args.get("mouse_actions") if extra_args else None
        keyboard_actions = extra_args.get("keyboard_actions") if extra_args else None

        if mouse_actions is None or keyboard_actions is None:
            if num_action_frames is not None:
                # Warmup path: supply zero-filled no-op actions so the model
                # can still trace its forward pass.
                mouse_actions = torch.zeros((num_action_frames, 2), dtype=dtype, device=device)
                keyboard_actions = torch.zeros((num_action_frames, 7), dtype=dtype, device=device)
            else:
                raise ValueError("Both 'mouse_actions' and 'keyboard_actions' must be provided in extra_params.")

        # Convert to tensors if needed
        if not isinstance(mouse_actions, torch.Tensor):
            mouse_actions = torch.tensor(mouse_actions, dtype=dtype, device=device)
        else:
            mouse_actions = mouse_actions.to(dtype=dtype, device=device)

        if not isinstance(keyboard_actions, torch.Tensor):
            keyboard_actions = torch.tensor(keyboard_actions, dtype=dtype, device=device)
        else:
            keyboard_actions = keyboard_actions.to(dtype=dtype, device=device)

        # Ensure batch dimension
        if mouse_actions.ndim == 2:
            mouse_actions = mouse_actions.unsqueeze(0)
        if keyboard_actions.ndim == 2:
            keyboard_actions = keyboard_actions.unsqueeze(0)

        return mouse_actions, keyboard_actions

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        mouse_actions: torch.Tensor | None = None,
        keyboard_actions: torch.Tensor | None = None,
        action_context_scale: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through transformer to predict noise."""
        if current_model is None:
            current_model = self.transformer
        return current_model(
            mouse_actions=mouse_actions,
            keyboard_actions=keyboard_actions,
            action_context_scale=action_context_scale,
            **kwargs,
        )[0]

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        frame_num: int = 49,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Extract prompt from request
        if len(req.prompts) > 1:
            raise ValueError("This model only supports a single prompt.")
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale

        # Ensure dimensions are compatible
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype

        # Generator
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Parse actions (num_action_frames = temporal_ratio * latent_frames + 1)
        extra_args = req.sampling_params.extra_args if hasattr(req.sampling_params, "extra_args") else None
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        num_action_frames = num_latent_frames * self.vae_scale_factor_temporal + 1
        mouse_actions, keyboard_actions = self._parse_actions(
            extra_args, device, dtype, num_action_frames=num_action_frames
        )

        # Encode prompt
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                device=device,
                dtype=dtype,
            )

        # Prepare latents
        num_channels_latents = self.transformer_config.in_channels
        latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        # Denoising loop
        with self.progress_bar(total=len(timesteps)) as pbar:
            for t in timesteps:
                self._current_timestep = t
                latent_model_input = latents.to(dtype)
                timestep = t.expand(latents.shape[0])

                # Duplicate actions for CFG
                if do_cfg:
                    mouse_actions_cfg = torch.cat([mouse_actions] * 2)
                    keyboard_actions_cfg = torch.cat([keyboard_actions] * 2)
                else:
                    mouse_actions_cfg = mouse_actions
                    keyboard_actions_cfg = keyboard_actions

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "return_dict": False,
                    "current_model": self.transformer,
                    "mouse_actions": mouse_actions_cfg if not do_cfg else mouse_actions,
                    "keyboard_actions": keyboard_actions_cfg if not do_cfg else keyboard_actions,
                }
                if do_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "return_dict": False,
                        "current_model": self.transformer,
                        "mouse_actions": mouse_actions,
                        "keyboard_actions": keyboard_actions,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_cfg)
                pbar.update()

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        # Decode
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )
