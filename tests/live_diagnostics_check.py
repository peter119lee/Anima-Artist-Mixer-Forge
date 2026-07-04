"""Live validation for the v27.1 diagnostics nodes (TagCheck / ABVariants /
ImpactMap) against a running ComfyUI with the real Anima model.

Manual harness (GPU + live server required):

    python tests/live_diagnostics_check.py
"""

import json
import os
import time
import urllib.error
import urllib.request

SERVER = os.environ.get("ANIMA_SMOKE_SERVER", "http://127.0.0.1:8188")

UNET = "Anima\\anime\\anima_baseV10.safetensors"
CLIP = "qwen_3_06b_base.safetensors"
VAE = "qwen_image_vae.safetensors"

BASE_PROMPT = (
    "1girl, solo, masterpiece, best quality, upper body portrait, face visible, "
    "wearing a white blouse and navy jacket, looking at viewer, simple background"
)
NEG_PROMPT = "nsfw, nude, lowres, worst quality, bad anatomy"
SEED = 20260704


def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path, data=data, headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=30))


def _get(path):
    return json.load(urllib.request.urlopen(SERVER + path, timeout=30))


def submit_and_wait(name, graph, timeout=420):
    t0 = time.time()
    try:
        resp = _post("/prompt", {"prompt": graph})
    except urllib.error.HTTPError as e:
        return name, "SUBMIT_FAIL", e.read().decode("utf-8", "replace")[:800], {}
    if resp.get("node_errors"):
        return name, "NODE_ERRORS", json.dumps(resp["node_errors"])[:800], {}
    pid = resp["prompt_id"]
    while time.time() - t0 < timeout:
        hist = _get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                detail = json.dumps(entry["status"].get("messages", []))[:800]
                return name, "EXEC_ERROR", detail, {}
            return name, "OK", f"{time.time() - t0:.1f}s", entry.get("outputs", {})
        time.sleep(2)
    return name, "TIMEOUT", f">{timeout}s", {}


def loaders():
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "stable_diffusion",
                         "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG_PROMPT, "clip": ["2", 0]}},
    }


def tagcheck_graph(chain):
    g = loaders()
    g["4"] = {"class_type": "AnimaArtistPack",
              "inputs": {"clip": ["2", 0], "artist_chain": chain,
                         "base_prompt": BASE_PROMPT}}
    g["20"] = {"class_type": "AnimaArtistTagCheck",
               "inputs": {"artist_pack": ["4", 0]}}
    return g


def preset_apply(g, pack_id, node_id):
    g["5"] = {"class_type": "AnimaArtistPreset",
              "inputs": {"preset": "balanced", "intensity": 1.0,
                         "normalize_weights": True, "layer_mode": "auto",
                         "custom_layer_filter": ""}}
    g[node_id] = {"class_type": "AnimaArtistPresetApply",
                  "inputs": {"model": ["1", 0], "artist_pack": [pack_id, 0],
                             "preset": ["5", 0], "enabled": True,
                             "apply_to_uncond": False}}


def sampler(g, node_id, apply_id, width=512, height=512, steps=8):
    g["8" if node_id == "9" else f"lat{node_id}"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1}}
    latent = "8" if node_id == "9" else f"lat{node_id}"
    g[node_id] = {"class_type": "KSampler",
                  "inputs": {"model": [apply_id, 0], "positive": [apply_id, 1],
                             "negative": ["7", 0], "latent_image": [latent, 0],
                             "seed": SEED, "steps": steps, "cfg": 5.0,
                             "sampler_name": "er_sde", "scheduler": "beta",
                             "denoise": 1.0}}


def ab_variants_graph():
    g = loaders()
    g["30"] = {"class_type": "AnimaArtistABVariants",
               "inputs": {"artist_chain": "@uof, @kieed", "mode": "solo_each",
                          "include_no_mixer": True, "include_full_mix": True}}
    g["4"] = {"class_type": "AnimaArtistPack",
              "inputs": {"clip": ["2", 0], "artist_chain": ["30", 0],
                         "base_prompt": BASE_PROMPT}}
    preset_apply(g, "4", "6")
    sampler(g, "9", "6")
    g["10"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["3", 0]}}
    g["11"] = {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": ["30", 1]}}
    return g


def impact_map_graph(identical=False):
    g = loaders()
    # Branch A: no mixer (empty chain still encodes the base prompt).
    g["40"] = {"class_type": "AnimaArtistPack",
               "inputs": {"clip": ["2", 0], "artist_chain": "",
                          "base_prompt": BASE_PROMPT}}
    preset_apply(g, "40", "41")
    sampler(g, "42", "41")
    g["43"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["42", 0], "vae": ["3", 0]}}
    # Branch B: mixer on.
    g["50"] = {"class_type": "AnimaArtistPack",
               "inputs": {"clip": ["2", 0], "artist_chain": "@uof, @kieed",
                          "base_prompt": BASE_PROMPT}}
    g["51"] = {"class_type": "AnimaArtistPresetApply",
               "inputs": {"model": ["1", 0], "artist_pack": ["50", 0],
                          "preset": ["5", 0], "enabled": True,
                          "apply_to_uncond": False}}
    sampler(g, "52", "51")
    g["53"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["52", 0], "vae": ["3", 0]}}
    a_src = ["53", 0] if identical else ["43", 0]
    g["60"] = {"class_type": "AnimaArtistImpactMap",
               "inputs": {"image_a": a_src, "image_b": ["53", 0],
                          "layout": "triptych", "auto_gain": True, "gain": 4.0}}
    g["61"] = {"class_type": "SaveImage",
               "inputs": {"images": ["60", 0],
                          "filename_prefix": "diag_impact_identical"
                          if identical else "diag_impact_off_vs_on"}}
    return g


def texts_of(outputs, node_id):
    out = outputs.get(node_id) or {}
    txt = out.get("text")
    if isinstance(txt, list):
        return "\n".join(str(t) for t in txt)
    return str(txt) if txt else ""


def images_of(outputs):
    imgs = []
    for out in outputs.values():
        for img in out.get("images", []):
            imgs.append(img["filename"])
    return imgs


def main():
    results = []

    name, status, detail, outputs = submit_and_wait(
        "tagcheck known+gibberish",
        tagcheck_graph("@uof, @kieed, @zzqqxnotanartist9999"))
    report = texts_of(outputs, "20")
    ok = status == "OK" and "dist" in report
    results.append((name, "OK" if ok else status or "NO_REPORT", detail))
    print(f"--- {name}: {status} {detail}\n{report}\n")

    name, status, detail, outputs = submit_and_wait(
        "tagcheck duplicate", tagcheck_graph("@uof, @uof"))
    report = texts_of(outputs, "20")
    ok = status == "OK" and "[DUPLICATE]" in report
    results.append((name, "OK" if ok else "MISSING_DUP_FLAG", detail))
    print(f"--- {name}: {status} {detail}\n{report}\n")

    name, status, detail, outputs = submit_and_wait(
        "ab_variants fan-out x4", ab_variants_graph(), timeout=600)
    imgs = images_of(outputs)
    report = texts_of(outputs, "30")
    ok = status == "OK" and len(imgs) == 4
    results.append((name, "OK" if ok else f"IMG_COUNT_{len(imgs)}", detail))
    print(f"--- {name}: {status} {detail}\n  images: {imgs}\n{report}\n")

    name, status, detail, outputs = submit_and_wait(
        "impact_map off_vs_on", impact_map_graph(identical=False), timeout=600)
    report = texts_of(outputs, "60")
    imgs = images_of(outputs)
    ok = status == "OK" and "impact" in report and imgs
    results.append((name, "OK" if ok else status, detail))
    print(f"--- {name}: {status} {detail}\n  images: {imgs}\n{report}\n")

    name, status, detail, outputs = submit_and_wait(
        "impact_map identical", impact_map_graph(identical=True), timeout=600)
    report = texts_of(outputs, "60")
    ok = status == "OK" and "no visible change" in report
    results.append((name, "OK" if ok else "UNEXPECTED_REPORT", detail))
    print(f"--- {name}: {status} {detail}\n{report}\n")

    print("== summary ==")
    failures = 0
    for name, status, detail in results:
        print(f"  {status:18s} {name}")
        failures += status != "OK"
    print(f"== {failures} failure(s) ==")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
