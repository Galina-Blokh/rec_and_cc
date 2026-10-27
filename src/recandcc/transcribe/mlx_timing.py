# Vendored from mlx-whisper 0.4.3, mlx_whisper/timing.py (WordTiming, find_alignment,
# merge_punctuations, add_word_timestamps), with the numba/scipy kernels replaced by
# recandcc.transcribe._align_np and the alignment heads capped (capped_heads); changes
# are marked "RecAndCC".
#
# Copyright (c) 2022 OpenAI (Whisper, https://github.com/openai/whisper)
# Copyright © 2023 Apple Inc. (mlx-examples, https://github.com/ml-explore/mlx-examples)
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""Word timestamps for MLX Whisper without numba or scipy.

``gpu.import_mlx_whisper`` registers this module as ``mlx_whisper.timing`` (only for
mlx-whisper 0.4.3, the version it was copied from), so ``mlx_whisper.transcribe`` picks up
``add_word_timestamps`` from here. mlx and mlx_whisper are imported lazily: this module is
importable on every platform, and importing mlx_whisper here would be circular.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._align_np import dtw, median_filter

# mlx_whisper.audio constants (not imported: see the module docstring).
HOP_LENGTH = 160
SAMPLE_RATE = 16000
TOKENS_PER_SECOND = SAMPLE_RATE // HOP_LENGTH // 2  # 50


# RecAndCC: mlx-whisper never loads tuned alignment heads, so every head of the decoder's last
# half is used (40 on turbo, 320 on large-v3). The median filter over all of them costs
# seconds per window while the GPU thread is held, so at most this many are kept.
MAX_ALIGNMENT_HEADS = 64


def capped_heads(heads: list[list[int]]) -> list[list[int]]:
    """``heads`` ((layer, head) pairs) unchanged up to MAX_ALIGNMENT_HEADS, else a fixed subset:
    ordered last layer first, then evenly spaced, so the latest layers are always included."""
    if len(heads) <= MAX_ALIGNMENT_HEADS:
        return heads
    ordered = sorted(heads, key=lambda lh: (-lh[0], lh[1]))
    picks = np.linspace(0, len(ordered) - 1, MAX_ALIGNMENT_HEADS).round().astype(int)
    return [ordered[i] for i in picks]


@dataclass
class WordTiming:
    word: str
    tokens: list[int]
    start: float
    end: float
    probability: float


def find_alignment(
    model: Any,
    tokenizer: Any,
    text_tokens: list[int],
    mel: Any,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> list[WordTiming]:
    if len(text_tokens) == 0:
        return []
    import mlx.core as mx  # pyright: ignore[reportMissingImports]

    tokens = mx.array(
        [*tokenizer.sot_sequence, tokenizer.no_timestamps, *text_tokens, tokenizer.eot]
    )
    logits, cross_qk = model.forward_with_cross_qk(mel[None, :], tokens[None, :])
    # consider only the logits associated with predicting text
    sampled_logits = logits[0][len(tokenizer.sot_sequence) : -2, : tokenizer.eot]
    token_probs = mx.softmax(sampled_logits, precise=True, axis=-1)
    text_token_probs = mx.take_along_axis(
        token_probs, mx.array(text_tokens)[:, None], axis=1
    ).squeeze(1)
    text_token_probs = np.array(text_token_probs)

    # heads * tokens * frames
    heads = capped_heads(model.alignment_heads.tolist())  # RecAndCC: cost cap
    weights = mx.stack([cross_qk[_l][0, _h] for _l, _h in heads])
    weights = weights[:, :, : num_frames // 2]
    weights = mx.softmax(weights * qk_scale, axis=-1, precise=True)
    weights = weights.astype(mx.float32)
    mean = mx.mean(weights, axis=-2, keepdims=True)
    std = mx.var(weights, axis=-2, keepdims=True, ddof=0).sqrt()
    weights = (weights - mean) / std
    weights = median_filter(np.array(weights), medfilt_width)

    matrix = weights.mean(axis=0)
    matrix = matrix[len(tokenizer.sot_sequence) : -1]
    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        # eot only: np.pad([], (1, 0)) is a float array, which can't index jump_times
        return []
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:], strict=False)
    ]

    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities, strict=False
        )
    ]


def merge_punctuations(alignment: list[WordTiming], prepended: str, appended: str) -> None:
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            # prepend it to the following word
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in appended:
            # append it to the previous word
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1


def add_word_timestamps(
    *,
    segments: list[dict],
    model: Any,
    tokenizer: Any,
    mel: Any,
    num_frames: int,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    last_speech_timestamp: float,
    **kwargs: Any,
) -> None:
    if len(segments) == 0:
        return

    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot] for segment in segments
    ]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)
    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]
    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2

    # hack: truncate long words at sentence boundaries.
    # a better segmentation algorithm based on VAD should be able to replace this.
    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        # ensure words at sentence boundaries are not longer than twice the median word duration.
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks:
                    alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment, strict=False):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + timing.start, 2),
                        end=round(time_offset + timing.end, 2),
                        probability=float(timing.probability),
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # hack: truncate long words at segment boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(words) > 0:
            # ensure the first and second word after a pause is not longer than
            # twice the median word duration.
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (len(words) > 1 and words[1]["end"] - words[0]["start"] > max_duration * 2)
            ):
                if len(words) > 1 and words[1]["end"] - words[1]["start"] > max_duration:
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            # prefer the segment-level start timestamp if the first word is too long.
            if segment["start"] < words[0]["end"] and segment["start"] - 0.5 > words[0]["start"]:
                words[0]["start"] = max(0, min(words[0]["end"] - median_duration, segment["start"]))
            else:
                segment["start"] = words[0]["start"]

            # prefer the segment-level end timestamp if the last word is too long.
            if segment["end"] > words[-1]["start"] and segment["end"] + 0.5 < words[-1]["end"]:
                words[-1]["end"] = max(words[-1]["start"] + median_duration, segment["end"])
            else:
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words
