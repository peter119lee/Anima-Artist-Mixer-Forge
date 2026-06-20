"""
Automated test script for Anima-Artist-Mixer
Tests different configurations and generates comparison images
"""

import json
import time
import requests
import uuid
from pathlib import Path
from typing import Dict, Any, List

# Configuration
COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test seed for consistency
TEST_SEED = 12345

# Test configurations
TESTS = {
    "baseline": {
        "name": "1a_no_mixer_single",
        "description": "Single artist in prompt (baseline)",
        "use_mixer": False,
        "prompt": "yuchi (salmon-1000), a girl, simple background",
        "artist_chain": "",
    },
    "mixer_single_balanced": {
        "name": "1b_mixer_single_balanced",
        "description": "Single artist via mixer (balanced preset)",
        "use_mixer": True,
        "prompt": "a girl, simple background",
        "artist_chain": "@yuchi (salmon-1000)",
        "preset": "balanced",
    },
    "no_mixer_multi": {
        "name": "2a_no_mixer_multi",
        "description": "Multi-artist in prompt (shows interference)",
        "use_mixer": False,
        "prompt": "wlop, sakimichan, krenz, a girl, simple background",
        "artist_chain": "",
    },
    "mixer_multi_balanced": {
        "name": "2b_mixer_multi_balanced",
        "description": "Multi-artist via mixer (balanced)",
        "use_mixer": True,
        "prompt": "a girl, simple background",
        "artist_chain": "@wlop, @sakimichan, @krenz",
        "preset": "balanced",
    },
    "preset_strong_style": {
        "name": "3b_preset_strong_style",
        "description": "Preset: strong_style (strength 1.8)",
        "use_mixer": True,
        "prompt": "a girl, simple background",
        "artist_chain": "@wlop, @sakimichan, @krenz",
        "preset": "strong_style",
    },
    "preset_stable_seed": {
        "name": "3c_preset_stable_seed",
        "description": "Preset: stable_seed (cross-seed stability)",
        "use_mixer": True,
        "prompt": "a girl, simple background",
        "artist_chain": "@wlop, @sakimichan, @krenz",
        "preset": "stable_seed",
    },
    "preset_drift_auto": {
        "name": "3d_preset_drift_auto",
        "description": "Preset: drift_auto (automatic routing)",
        "use_mixer": True,
        "prompt": "a girl, simple background",
        "artist_chain": "@wlop, @sakimichan, @krenz",
        "preset": "drift_auto",
    },
    "preset_face_lock": {
        "name": "3e_preset_face_lock",
        "description": "Preset: face_lock (preserve facial details)",
        "use_mixer": True,
        "prompt": "a girl, close up, face focus",
        "artist_chain": "@wlop, @sakimichan, @krenz",
        "preset": "face_lock",
    },
    "preset_scene_lock": {
        "name": "3f_preset_scene_lock",
        "description": "Preset: scene_lock (protect background)",
        "use_mixer": True,
        "prompt": "a girl, wide shot, cityscape background",
        "artist_chain": "@wlop, @sakimichan, @krenz",
        "preset": "scene_lock",
    },
}


def check_comfyui_status():
    """Check if ComfyUI is running"""
    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        return response.status_code == 200
    except:
        return False


def convert_ui_to_api_format(ui_workflow: Dict) -> Dict:
    """Convert UI export format to API format"""
    api_workflow = {}

    for node in ui_workflow.get("nodes", []):
        node_id = str(node.get("id"))
        node_type = node.get("type", "")

        api_node = {
            "class_type": node_type,
            "inputs": {}
        }

        # Add widget values as inputs
        if "widgets_values" in node:
            # Widget values go into inputs with generic names
            # This is a simplified conversion - may need adjustment
            for i, value in enumerate(node["widgets_values"]):
                api_node["inputs"][f"widget_{i}"] = value

        # Add connections from links (simplified)
        if "inputs" in node:
            for inp in node["inputs"]:
                if "link" in inp and inp["link"]:
                    api_node["inputs"][inp.get("name", f"input_{inp.get('link')}")] = [
                        str(inp["link"]), 0  # [node_id, output_index]
                    ]

        api_workflow[node_id] = api_node

    return api_workflow


def load_workflow_template():
    """Load the test workflow as template"""
    workflow_path = Path("I:/ComfyUI-aki-v1.6/ComfyUI/user/default/workflows/test_multi_sampler.json")
    with open(workflow_path, 'r', encoding='utf-8') as f:
        ui_workflow = json.load(f)

    # Check if it's UI format
    if "nodes" in ui_workflow and "links" in ui_workflow:
        print("⚠ Converting from UI format to API format...")
        return ui_workflow  # Return UI format, we'll handle it differently
    else:
        return ui_workflow


def modify_workflow(template: Dict, test_config: Dict) -> Dict:
    """Modify workflow based on test configuration"""
    workflow = json.loads(json.dumps(template))  # Deep copy

    # Find and modify relevant nodes
    for node in workflow.get("nodes", []):
        node_type = node.get("type", "")

        # Modify seed
        if node_type == "Seed (rgthree)" or "seed" in str(node.get("widgets_values", [])).lower():
            if "widgets_values" in node:
                for i, val in enumerate(node["widgets_values"]):
                    if isinstance(val, int) and val > 1000:  # Likely a seed
                        node["widgets_values"][i] = TEST_SEED

        # Modify AnimaArtistPack
        if node_type == "AnimaArtistPack":
            widgets = node.get("widgets_values", [])
            if len(widgets) >= 2:
                widgets[0] = test_config.get("artist_chain", "")  # artist_chain
                widgets[1] = test_config.get("prompt", "")  # base_prompt

        # Modify AnimaArtistPreset
        if node_type == "AnimaArtistPreset":
            widgets = node.get("widgets_values", [])
            if len(widgets) >= 1:
                preset = test_config.get("preset", "balanced")
                widgets[0] = preset  # preset parameter

        # Modify CLIPTextEncode for non-mixer tests
        if node_type == "CLIPTextEncode" and not test_config.get("use_mixer", True):
            widgets = node.get("widgets_values", [])
            if len(widgets) >= 1:
                widgets[0] = test_config.get("prompt", "")

        # Modify SaveImage to use unique filename
        if node_type == "SaveImage":
            widgets = node.get("widgets_values", [])
            if len(widgets) >= 1:
                widgets[0] = test_config.get("name", "test")  # filename_prefix

    return workflow


def queue_prompt(workflow: Dict) -> str:
    """Queue a prompt and return the prompt_id"""
    prompt_data = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4())
    }

    response = requests.post(
        f"{COMFYUI_URL}/prompt",
        json=prompt_data
    )

    if response.status_code == 200:
        result = response.json()
        return result.get("prompt_id")
    else:
        raise Exception(f"Failed to queue prompt: {response.status_code} {response.text}")


def wait_for_completion(prompt_id: str, timeout: int = 300):
    """Wait for a prompt to complete"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("completed", False):
                        return True, "completed"
                    elif "error" in status:
                        return False, status.get("error", "Unknown error")
        except Exception as e:
            print(f"Error checking status: {e}")

        time.sleep(2)

    return False, "timeout"


def run_test(test_id: str, test_config: Dict, template: Dict):
    """Run a single test"""
    print(f"\n{'='*60}")
    print(f"Running: {test_config['name']}")
    print(f"Description: {test_config['description']}")
    print(f"{'='*60}")

    # Modify workflow
    workflow = modify_workflow(template, test_config)

    # Queue prompt
    try:
        prompt_id = queue_prompt(workflow)
        print(f"✓ Queued (prompt_id: {prompt_id})")
    except Exception as e:
        print(f"✗ Failed to queue: {e}")
        return False

    # Wait for completion
    print("Waiting for generation...")
    success, result = wait_for_completion(prompt_id)

    if success:
        print(f"✓ Completed successfully")
        return True
    else:
        print(f"✗ Failed: {result}")
        return False


def generate_report(results: Dict[str, bool]):
    """Generate a test report"""
    report_path = OUTPUT_DIR / "test_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Anima-Artist-Mixer Test Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Seed: {TEST_SEED}\n")
        f.write(f"Total Tests: {len(results)}\n")
        f.write(f"Passed: {sum(results.values())}\n")
        f.write(f"Failed: {len(results) - sum(results.values())}\n\n")

        f.write("Test Results:\n")
        f.write("-" * 60 + "\n")
        for test_id, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            test_config = TESTS.get(test_id, {})
            f.write(f"{status} | {test_config.get('name', test_id)}\n")
            f.write(f"       {test_config.get('description', '')}\n\n")

        f.write("\nNext Steps:\n")
        f.write("-" * 60 + "\n")
        f.write("1. Check generated images in ComfyUI output folder\n")
        f.write("2. Compare images visually\n")
        f.write("3. Document findings\n")

    print(f"\n✓ Report saved to: {report_path}")


def main():
    print("Anima-Artist-Mixer Automated Test")
    print("=" * 60)

    # Check ComfyUI status
    print("\nChecking ComfyUI status...")
    if not check_comfyui_status():
        print("✗ ComfyUI is not running!")
        print("Please start ComfyUI first:")
        print("  cd I:\\ComfyUI-aki-v1.6\\ComfyUI")
        print("  python main.py --listen 127.0.0.1 --port 8188")
        return

    print("✓ ComfyUI is running")

    # Load workflow template
    print("\nLoading workflow template...")
    try:
        template = load_workflow_template()
        print("✓ Template loaded")
    except Exception as e:
        print(f"✗ Failed to load template: {e}")
        return

    # Run tests
    results = {}

    for test_id, test_config in TESTS.items():
        success = run_test(test_id, test_config, template)
        results[test_id] = success

        # Wait a bit between tests
        time.sleep(3)

    # Generate report
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    generate_report(results)

    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"\nSummary: {passed}/{total} tests passed")

    if passed == total:
        print("✓✓✓ All tests passed!")
    else:
        print(f"⚠ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
