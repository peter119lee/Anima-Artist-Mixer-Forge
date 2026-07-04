# Development guide

How to add a new node to this pack and ship it to the ComfyUI registry.

## Repo layout

- `anima_mixer/` — the real package.
  - `nodes_core.py` — the model-patching engine nodes (`AnimaArtistCrossAttn`,
    `AnimaArtistPresetApply`) and the shared runtime-state builder.
  - `nodes_pack.py` — `AnimaArtistBasic` and `AnimaArtistPack` (encoding + packing).
  - `nodes_probe.py` — `AnimaArtistProbe` / `AnimaArtistProbeReport`.
  - `nodes_ui.py` — chain building / preview / preset / starter / inspector nodes.
  - `nodes_options.py` — `AnimaArtistOptions` / `AnimaArtistSimpleOptions`.
  - `nodes_recipes.py` — `AnimaArtistRecipeSave` / `AnimaArtistRecipeLoad`.
  - `nodes_diagnostics.py` — `AnimaArtistTagCheck` / `AnimaArtistABVariants` /
    `AnimaArtistImpactMap`.
  - `wrapper.py` — the cross-attention forward wrapper (dispatch + fusion math).
  - `wrapper_stabilizers.py` — `StabilizerMixin` (EMA, static capture, norm
    lock, delta cap, anchor-Q).
  - `patching.py`, `anchor.py`, `parsing.py`, `chain_tools.py`, `options.py`,
    `recipe.py`, `constants.py`, `math_utils.py`, `tag_vocab.py`,
    `probe_stats.py` — supporting modules.
  - Compat contract: `nodes_core` re-exports the pack/probe node classes and
    `nodes_ui` re-exports the options/recipes node classes, so pre-v27.4
    import paths keep working. Keep every module under 800 lines.
- `__init__.py` / `nodes.py` — ComfyUI entry point and a compatibility shim;
  both re-export the mappings from `anima_mixer/__init__.py`.
- `tests/` — the pytest suite CI runs. `tests/live_*.py` are manual scripts
  that need a running ComfyUI at `http://127.0.0.1:8188`.
- `workflow/` and `sample workflow.json` — bundled workflows, validated by
  `tests/test_workflow_json.py`.
- `anima_mixer/data/danbooru_tags.csv.gz` — bundled Danbooru tag metadata
  used by the ChainPreview/TagCheck vocabulary check; rebuild with
  `tools/build_tag_vocab.py` (provenance in its docstring). Keep the
  snapshot roughly aligned with the base model's training cutoff — newer is
  not automatically better.

## Adding a node

1. Implement the class in the `anima_mixer/nodes_*.py` module whose nodes it
   belongs with (see the layout above; e.g. diagnostics nodes go in
   `nodes_diagnostics.py`), following the existing `INPUT_TYPES` /
   `RETURN_TYPES` / `FUNCTION` / `CATEGORY` pattern used by its neighbors.
2. Register it in `anima_mixer/__init__.py`: add the import, a
   `NODE_CLASS_MAPPINGS` entry, and a `NODE_DISPLAY_NAME_MAPPINGS` entry.
   - Keep the `AnimaArtist` id prefix.
   - Do not reuse a node id that upstream Anima-Artist-Mixer defines unless it
     is intentionally the same node: ComfyUI registers node ids globally, so
     duplicate ids across installed packs shadow each other.
3. Add tests under `tests/`. Existing regression suites are
   `unittest.TestCase` style; plain pytest style also works (CI runs pytest).
4. If the node appears in a bundled workflow, remember that
   `tests/test_workflow_json.py` checks every tracked workflow's Anima nodes
   against `INPUT_TYPES` (widget counts, node-type whitelist, link refs).
   Changing a node's widgets means re-saving the bundled workflows that use
   it — that test failing on a widget change is intentional.

## Conventions

- Lint: `python -m ruff check .` (line length 110, `E501` ignored).
- Repo style: no type annotations, module-level `logger`, English messages.
- Runtime state lives in the shared `state` dict owned by the patcher; read
  the v26.2.0 CHANGELOG entry before touching reset or cache-keying logic
  (run-start reset, forward fingerprints, and CFG row selection are load-bearing).

## Verifying

```
python -m pytest -q      # full suite
python -m ruff check .
```

## Releasing to the ComfyUI registry

1. Bump `[project] version` in `pyproject.toml`.
2. Add a matching `## v<version> (<date>)` entry at the top of
   `CHANGELOG.md` — the publish workflow fails if the two versions differ.
3. Push to `main`. `.github/workflows/publish.yml` triggers on pushes that
   touch `pyproject.toml` (or run it manually via workflow_dispatch) and
   publishes to the registry as `anima-artist-mixer-forge`.
4. Publishing requires the `REGISTRY_ACCESS_TOKEN` repository secret: an API
   key for the `peter119lee` publisher created at
   <https://registry.comfy.org>. The `PublisherId` in `pyproject.toml` must
   match that publisher id.
