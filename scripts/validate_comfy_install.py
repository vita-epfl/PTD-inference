# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab
# SPDX-License-Identifier: Apache-2.0
"""Check the real ComfyUI node loader and example connections without weights.

Install this repository first, then supply the pinned ComfyUI checkout. This
CPU smoke check uses ComfyUI's own node registration, schemas and socket type
validation. It does not load models, start a server or validate model filenames.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from pathlib import Path
import sys


MODEL_SELECTORS = {
    ("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
    ("VAELoader", "vae_name"), ("PTDH3Loader", "checkpoint"),
}


async def validate(comfy_root: Path, repository: Path) -> None:
    sys.path.insert(0, str(comfy_root))
    import comfy.cli_args

    # Set this before importing nodes/server/model_management on CPU-only CI.
    comfy.cli_args.args.cpu = True
    import nodes
    from comfy_execution.validation import validate_node_input

    # Load only the real built-ins referenced by the shipped example.
    for filename in ("nodes_custom_sampler.py", "nodes_minimax_h3.py", "nodes_lt.py",
                     "nodes_audio.py", "nodes_video.py"):
        path = comfy_root / "comfy_extras" / filename
        if not await nodes.load_custom_node(str(path), module_parent="comfy_extras"):
            raise RuntimeError(f"ComfyUI could not register {filename}")
    if not await nodes.load_custom_node(str(repository)):
        raise RuntimeError("ComfyUI could not register the PTD custom-node package")
    expected = {"PTDH3Loader", "PTDH3Sampler", "PTDH3Sigmas"}
    for name in expected:
        cls = nodes.NODE_CLASS_MAPPINGS[name]
        if not Path(inspect.getfile(cls)).resolve().is_relative_to(repository):
            raise RuntimeError(f"{name} was imported from a different installation")

    workflow = json.loads((repository / "examples/comfyui/ptd_h3_t2v.json").read_text())
    by_id = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    if len(by_id) != len(workflow["nodes"]) or len(links) != len(workflow["links"]):
        raise ValueError("The workflow contains duplicate node or link IDs")
    schemas = {}
    for node_id, node in by_id.items():
        if node["type"] == "Note":  # Frontend-only annotation, with no sockets.
            if node.get("inputs") or node.get("outputs"):
                raise ValueError("A workflow Note cannot have runtime connections")
            continue
        cls = nodes.NODE_CLASS_MAPPINGS[node["type"]]
        if hasattr(cls, "GET_NODE_INFO_V1"):
            info = cls.GET_NODE_INFO_V1()
            inputs, outputs = info["input"], info["output"]
        else:
            inputs, outputs = cls.INPUT_TYPES(), cls.RETURN_TYPES
        sockets = {**inputs.get("required", {}), **inputs.get("optional", {})}
        schemas[node_id] = (sockets, outputs)
        if len(node.get("outputs", [])) != len(outputs):
            raise ValueError(f"Output schema changed for {node['type']}")
        widgets = dict(node.get("widgets_values_named", {}))
        if any(len(spec) > 1 and spec[1].get("control_after_generate") for spec in sockets.values()):
            # This companion seed widget belongs to the frontend, not execute().
            widgets.pop("control_after_generate", None)
        provided = set(widgets)
        for socket in node.get("inputs", []):
            name = socket["name"]
            if name not in sockets or not validate_node_input(socket["type"], sockets[name][0], strict=True):
                raise ValueError(f"Unknown or mistyped input {node['type']}.{name}")
            if socket.get("link") is not None:
                if socket["link"] not in links:
                    raise ValueError(f"Missing link for {node['type']}.{name}")
                provided.add(name)
        if not set(inputs.get("required", {})).issubset(provided):
            raise ValueError(f"Missing required inputs/widgets for {node['type']}")
        for name, value in widgets.items():
            if name not in sockets:
                raise ValueError(f"Unknown widget {node['type']}.{name}")
            options = sockets[name][0]
            # Model selectors depend on local files (VAELoader also includes
            # built-in entries). CI checks their socket/schema, not file access.
            if ((node["type"], name) not in MODEL_SELECTORS
                    and isinstance(options, list) and value not in options):
                raise ValueError(f"Unknown widget choice for {node['type']}.{name}")

    for link_id, source_id, source_slot, target_id, target_slot, declared_type in links.values():
        source, target = by_id[source_id], by_id[target_id]
        output = source["outputs"][source_slot]
        socket = target["inputs"][target_slot]
        source_type = schemas[source_id][1][source_slot]
        target_type = schemas[target_id][0][socket["name"]][0]
        if socket["link"] != link_id or link_id not in output.get("links", []):
            raise ValueError(f"Inconsistent workflow endpoints for link {link_id}")
        if not all(validate_node_input(source_type, value, strict=True)
                   for value in (target_type, declared_type, output["type"])):
            raise ValueError(f"Workflow link {link_id} disagrees with the runtime schema")
    for node in by_id.values():
        for index, output in enumerate(node.get("outputs", [])):
            actual = {link[0] for link in links.values() if link[1:3] == [node["id"], index]}
            if actual != set(output.get("links") or []):
                raise ValueError(f"Orphan output links on {node['type']}")
    print(json.dumps({"registered": sorted(expected), "workflow_nodes": len(by_id),
                      "workflow_links": len(links), "model_assets_loaded": False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(validate(args.comfy_root.resolve(), Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    main()
