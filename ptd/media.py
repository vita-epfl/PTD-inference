# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MP4 writing (H.264 video, optional stereo AAC audio) with PyAV."""

from __future__ import annotations

import os
from fractions import Fraction
from typing import Optional

import torch


def save_mp4(path: str, video: torch.Tensor, fps: int, audio: Optional[torch.Tensor] = None, sample_rate: int = 32000) -> str:
    """``video [3, T, H, W]`` in [-1, 1]; ``audio [2, samples]`` in [-1, 1]."""
    import av

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = video.permute(1, 2, 3, 0).float().mul(0.5).add(0.5).clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    _, height, width, _ = frames.shape
    with av.open(path, mode="w") as container:
        v_stream = container.add_stream("libx264", rate=int(fps))
        v_stream.width, v_stream.height, v_stream.pix_fmt = width, height, "yuv420p"
        a_stream = None
        if audio is not None:
            a_stream = container.add_stream("aac", rate=sample_rate)
            a_stream.codec_context.sample_rate = sample_rate
            a_stream.codec_context.layout = "stereo"
            a_stream.codec_context.time_base = Fraction(1, sample_rate)
        for frame in frames:
            for packet in v_stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in v_stream.encode():
            container.mux(packet)
        if a_stream is not None:
            pcm = (audio.float().cpu().T.clamp(-1, 1) * 32767).to(torch.int16).contiguous().view(1, -1).numpy()
            frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="stereo")
            frame.sample_rate = sample_rate
            resampler = av.audio.resampler.AudioResampler(format=a_stream.codec_context.format,
                                                          layout=a_stream.codec_context.layout, rate=sample_rate)
            pts = 0
            for chunk in resampler.resample(frame):
                chunk.pts = pts
                pts += chunk.samples
                for packet in a_stream.encode(chunk):
                    container.mux(packet)
            for packet in a_stream.encode():
                container.mux(packet)
    return path
