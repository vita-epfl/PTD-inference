# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt lists and condition-cache keys.

Accepted sources: ``.json`` (a list of strings, or of objects with a ``"prompt"`` field), ``.jsonl`` (one string or
object per line), ``.txt`` (one prompt per line), ``.tar`` (every ``<key>.txt`` member, WebDataset style) and
directories of ``.tar`` shards.  JSON keeps prompts that span several lines (MiniMax-H3's multi-field prompts).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import tarfile
from typing import List, Sequence


def prompt_cache_key(prompt: str) -> str:
    """Key of a request in a condition cache: ``sha1(prompt.strip())``."""
    return hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()


def prompt_cache_path(cache_dir: str, prompt: str) -> str:
    key = prompt_cache_key(prompt)
    return os.path.join(cache_dir, key[:2], f"{key}.pt")


def _from_json_value(e) -> str:
    return e if isinstance(e, str) else e["prompt"]


def load_prompts(paths: Sequence[str]) -> List[str]:
    prompts: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for tar in sorted(glob.glob(os.path.join(path, "**", "*.tar"), recursive=True)):
                prompts += load_prompts([tar])
        elif path.endswith(".tar"):
            with tarfile.open(path) as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith(".txt"):
                        prompts.append(tf.extractfile(m).read().decode("utf-8"))
        elif path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            prompts += [_from_json_value(e) for e in (data.values() if isinstance(data, dict) else data)]
        elif path.endswith(".jsonl"):
            with open(path) as f:
                prompts += [_from_json_value(json.loads(line)) for line in f if line.strip()]
        else:
            with open(path) as f:
                prompts += [line for line in f]
    prompts = [p.strip() for p in prompts if p.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {list(paths)}")
    return prompts
