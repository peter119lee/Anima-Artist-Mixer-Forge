# Anima 节点用途展示 workflow

这个文件夹用来说明当前 15 个节点各自为什么存在，以及应该在什么场景使用。

这些文件是 **ComfyUI API workflow**。可以像 `workflow/pr4_self_test_api` 一样用 `/prompt` 提交运行，也可以只作为接线参考。

## 运行方式

先确认 ComfyUI 已启动。这些 workflow 使用占位模型文件名，运行前请把各 loader 改成本机实际文件名：

- UNET（`UNETLoader` 的 `unet_name`）: `anima-base-v1.0.safetensors`
- CLIP（`CLIPLoader` 的 `clip_name`）: `qwen_3_06b_base.safetensors`
- VAE（`VAELoader` 的 `vae_name`）: `qwen_image_vae.safetensors`

运行单个 workflow：

```powershell
$server = "http://127.0.0.1:8190"
$file = "workflow\node_usage_showcase\01_basic_recommended.json"
$wf = Get-Content -Raw $file | ConvertFrom-Json
$body = @{ prompt = $wf } | ConvertTo-Json -Depth 100
Invoke-RestMethod -Uri "$server/prompt" -Method Post -ContentType "application/json" -Body $body
```

如果你使用默认端口，把 `$server` 改成：

```powershell
$server = "http://127.0.0.1:8188"
```

## 每个 workflow 展示什么

| 文件 | 展示节点 | 用途 |
|---|---|---|
| `01_basic_recommended.json` | `AnimaArtistBasic` | 最简单的一节点推荐入口 |
| `02_clean_preset_apply_with_inspector.json` | `AnimaArtistPack`, `AnimaArtistPreset`, `AnimaArtistSimpleOptions`, `AnimaArtistPresetApply`, `AnimaArtistInspector` | 标准 preset 路线；不会显示被 preset 覆盖的手动参数 |
| `03_starter_guided_preset.json` | `AnimaArtistStarter`, `AnimaArtistPack`, `AnimaArtistPresetApply`, `AnimaArtistInspector` | 表格输入、多 artist、自动生成 guide 的入口 |
| `04_chain_builder_preview_manual_crossattn.json` | `AnimaArtistChainBuilder`, `AnimaArtistChainPreview`, `AnimaArtistOptions`, `AnimaArtistCrossAttn`, `AnimaArtistInspector` | 手动高级路线；需要直接控制 `combine_mode` / `fusion_mode` / `strength` 时使用 |
| `05_recipe_save_load_share.json` | `AnimaArtistPreset`, `AnimaArtistRecipeSave`, `AnimaArtistRecipeLoad`, `AnimaArtistPack`, `AnimaArtistPresetApply`, `AnimaArtistInspector` | 保存和载入可分享 recipe |
| `06_probe_and_report.json` | `AnimaArtistPack`, `AnimaArtistProbe`, `AnimaArtistProbeReport` | 测量 artist 风格主要影响哪些层 |

## 15 个节点的覆盖情况

| 节点 | 出现位置 |
|---|---|
| `AnimaArtistBasic` | `01_basic_recommended.json` |
| `AnimaArtistStarter` | `03_starter_guided_preset.json` |
| `AnimaArtistChainBuilder` | `04_chain_builder_preview_manual_crossattn.json` |
| `AnimaArtistChainPreview` | `04_chain_builder_preview_manual_crossattn.json` |
| `AnimaArtistSimpleOptions` | `02_clean_preset_apply_with_inspector.json` |
| `AnimaArtistPack` | `02`, `03`, `04`, `05`, `06` |
| `AnimaArtistPresetApply` | `02`, `03`, `05` |
| `AnimaArtistCrossAttn` | `04_chain_builder_preview_manual_crossattn.json` |
| `AnimaArtistOptions` | `04_chain_builder_preview_manual_crossattn.json` |
| `AnimaArtistPreset` | `02`, `05` |
| `AnimaArtistInspector` | `02`, `03`, `04`, `05` |
| `AnimaArtistRecipeSave` | `05_recipe_save_load_share.json` |
| `AnimaArtistRecipeLoad` | `05_recipe_save_load_share.json` |
| `AnimaArtistProbe` | `06_probe_and_report.json` |
| `AnimaArtistProbeReport` | `06_probe_and_report.json` |

## 设计原则

- 新手只需要 `AnimaArtistBasic`。
- 常规 preset 工作流使用 `AnimaArtistPack + AnimaArtistPreset + AnimaArtistPresetApply`。
- preset apply 节点不显示 `combine_mode`、`fusion_mode`、`strength`，因为这些值由 preset 决定。
- 需要手动控制这些值时，才使用 `AnimaArtistCrossAttn (Manual/Advanced)`。
- `Inspector`、`Probe`、`ProbeReport` 是调试和证明用节点，不是普通出图必需节点。
- `RecipeSave` / `RecipeLoad` 是分享配置用节点，不是普通出图必需节点。
