import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper

IGNORE_ID = -1


def load_wav(wav, target_sr, min_sr=16000):
    if not isinstance(wav, tuple):
        speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    else:
        speech, sample_rate = wav
        if isinstance(speech, np.ndarray):
            speech = torch.tensor([speech], dtype=torch.float32)

    if sample_rate != target_sr:
        assert sample_rate >= min_sr, f"wav sample rate {sample_rate} must be greater than {target_sr}"
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)

    speech = speech.to(dtype=torch.float32)
    return speech


def extract_speech_feat(prompt_wav, feat_extractor, device):
    speech = load_wav(prompt_wav, 24000)
    speech_feat = feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(device)
    speech_feat = speech_feat.unsqueeze(dim=0)
    speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(device)
    return speech_feat, speech_feat_len


def extract_speech_token(prompt_wav, speech_tokenizer_session, device):
    speech = load_wav(prompt_wav, 16000)
    assert speech.shape[1] / 16000 <= 30, "do not support extract speech token for audio longer than 30s"
    feat = whisper.log_mel_spectrogram(speech, n_mels=128)
    speech_token = (
        speech_tokenizer_session.run(
            None,
            {
                speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0]
        .flatten()
        .tolist()
    )
    speech_token = torch.tensor([speech_token], dtype=torch.int32).to(device)
    speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device)
    return speech_token, speech_token_len


def extract_spk_embedding(prompt_wav, campplus_session, device):
    speech = load_wav(prompt_wav, 16000)
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = (
        campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0]
        .flatten()
        .tolist()
    )
    embedding = torch.tensor([embedding]).to(device)
    return embedding


def extract_text_token(text, tokenizer, allowed_special):
    text_token = tokenizer.encode(text, allowed_special=allowed_special)
    text_token = torch.tensor([text_token], dtype=torch.int32)
    text_token_len = text_token.shape[1]
    return text_token, text_token_len


def concat_text_with_prompt_ids(
    text: torch.Tensor,
    text_len: torch.Tensor,
    prompt_text: torch.Tensor,
    prompt_text_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = torch.concat([prompt_text, text], dim=1)
    text_len = text_len + prompt_text_len
    return text, text_len


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
