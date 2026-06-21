# PR #4 真实测试指南

这个文件夹给你自己验证 PR 里的主要说法。你只需要读这个文件。

这些 workflow 是 **ComfyUI API 格式**，不是为了好看地摆在画布上，而是为了稳定复现测试。你可以用 PowerShell 直接提交到正在运行的 ComfyUI。

这些测试里，preset 路线都使用：

```text
AnimaArtistPack -> AnimaArtistPresetApply
AnimaArtistPreset -> AnimaArtistPresetApply
```

这样 `balanced`、`drift_auto`、`prompt_passthrough` 等 preset 会自己决定 `combine_mode`、`fusion_mode`、`strength`，界面上不会再显示那些会被 preset 覆盖的手动参数。只有你想手动调这些参数时，才使用 `AnimaArtistCrossAttn (Manual/Advanced)`。

## 测试前准备

1. 确认 ComfyUI 已启动，并且加载的是当前这个 PR 版本的 `Anima-Artist-Mixer`。
2. 确认 ComfyUI 服务器地址。下面示例默认是：

```powershell
$server = "http://127.0.0.1:8190"
```

如果你是默认端口，把它改成：

```powershell
$server = "http://127.0.0.1:8188"
```

3. 检查 workflow 里的模型文件名是否与你本机一致：

- UNET: `Anima\anime\anima_baseV10.safetensors`
- CLIP: `qwen_3_06b_base.safetensors`
- VAE: `qwen_image_vae.safetensors`

如果你的名字不同，直接改对应 JSON 里的 `UNETLoader`、`CLIPLoader`、`VAELoader`。

## 单个 workflow 运行方法

在仓库根目录运行：

```powershell
$server = "http://127.0.0.1:8190"
$file = "workflow\pr4_self_test_api\01_same_model_two_sampler.json"
$wf = Get-Content -Raw $file | ConvertFrom-Json
$body = @{ prompt = $wf } | ConvertTo-Json -Depth 100
Invoke-RestMethod -Uri "$server/prompt" -Method Post -ContentType "application/json" -Body $body
```

提交后去 ComfyUI 看 queue 和输出图片。输出文件名前缀都是 `anima_selftest_...`。

## 一次运行全部 workflow

```powershell
$server = "http://127.0.0.1:8190"
Get-ChildItem "workflow\pr4_self_test_api\*.json" | ForEach-Object {
  Write-Host "submit $($_.Name)"
  $wf = Get-Content -Raw $_.FullName | ConvertFrom-Json
  $body = @{ prompt = $wf } | ConvertTo-Json -Depth 100
  Invoke-RestMethod -Uri "$server/prompt" -Method Post -ContentType "application/json" -Body $body
}
```

建议先一个一个跑，确认没有模型路径问题后再全部跑。

## 每个 workflow 验证什么

### 01_same_model_two_sampler.json

验证 reviewer 报告的多 sampler 同模型问题。

这个 workflow 在同一个图里从同一个 UNET 分出两条采样分支：

- 一条是直接 prompt / no mixer
- 一条是 `AnimaArtistPack` + `AnimaArtistPreset(balanced)` + `AnimaArtistPresetApply`

期望结果：

- workflow 能跑完
- ComfyUI 不出现 `AttributeError: 'Attention' object has no attribute 'original'`
- 输出两张图，文件名前缀：
  - `anima_selftest_01_prompt_branch`
  - `anima_selftest_01_balanced_branch`

这项主要验证“同模型多 sampler 不再因为 `.original` patch/restore 失败”。

### 02_prompt_passthrough_direct_prompt.json

验证 `prompt_passthrough` 的用途。

这个 workflow 有两条分支：

- 直接 prompt：`(@yuchi \(salmon-1000\):1.2), (@uof:0.8), ...`
- `prompt_passthrough`：artist chain 写成 `1.2::@yuchi \(salmon-1000\)::, 0.8::@uof::`

期望结果：

- 两条分支都能跑完
- 两张图应该相同或几乎相同
- 输出文件名前缀：
  - `anima_selftest_02_direct_prompt`
  - `anima_selftest_02_prompt_passthrough`

这项验证“想走 no-mixer / 直接 prompt 路线，但保留 `1.2::tag::` 这种方便权重写法”应该使用 `prompt_passthrough`，不是 `balanced` 或 `drift_auto`。
它也验证 preset-only 节点能返回未 patch 的 model，所以速度应该接近直接 prompt；单次出图会受缓存、队列、VAE decode 等影响，不要把它当成严格 benchmark。

### 03_multi_artist_modes.json

验证多 artist 时不同模式的实际价值。

这个 workflow 对 4 个 artist 同 seed 跑三条分支：

- 直接 prompt / no mixer
- `balanced`，通过 `AnimaArtistPresetApply` 应用
- `drift_auto`，通过 `AnimaArtistPresetApply` 应用

artist chain:

```text
@yuchi \(salmon-1000\), @uof, @kieed, @ciloranko
```

期望结果：

- 三张图都能跑出来
- 三张图不应该完全一样
- `balanced` 是原始 mixer 风格路线
- `drift_auto` 是降低 style drift 的工程路线，不是 no-mixer / 直接 prompt 路线
- `balanced` 和 `drift_auto` 都不会在 preset apply 节点上显示 `combine_mode` / `fusion_mode` / `strength`

输出文件名前缀：

- `anima_selftest_03_prompt`
- `anima_selftest_03_balanced`
- `anima_selftest_03_drift_auto`

这项验证“PR 不是只加按钮，而是给多 artist 提供可控路线”。

### 04_routes_negative_timing.json

验证新语法能真实跑通。

这个 workflow 使用：

- 正/负 artist 权重
- layer route
- timing route
- timing fade
- `AnimaArtistPresetApply`，避免 preset 覆盖手动参数但界面仍显示手动参数的混淆

artist chain:

```text
1.0::@uof@0-12%0.0-0.55~0.10::
-0.5::@kieed@9-20%0.35-1.0~0.10::
@ciloranko@19-27
```

期望结果：

- workflow 能跑完
- 不出现解析错误或 runtime error
- 输出文件名前缀：
  - `anima_selftest_04_routes_negative_timing`

这项验证“负权重、layer route、timing route、fade 不是只在文档里写了，而是可以在真实采样中执行”。

## 怎么判断测试失败

如果失败，优先看这几类：

- `Cannot find class AnimaArtistPresetApply`：ComfyUI 没加载当前 PR 版本，重启 ComfyUI。
- `UNETLoader` / `CLIPLoader` / `VAELoader` 报找不到文件：改 JSON 里的模型文件名。
- `.original` 相关错误：确认没有加载旧版节点，确认 custom_nodes 里只有一个 `Anima-Artist-Mixer`。
- OOM：先把 JSON 里的 `width` / `height` 改成 `512`，`steps` 改成 `8`。

## 关于速度

这些 workflow 可以看大概耗时，但不要把单次运行时间当成严格 benchmark。ComfyUI queue、模型缓存、VAE decode、首次加载都会影响时间。

这里的测试目标是：

- 是否能真实跑完
- 是否修复同模型多 sampler 问题
- `prompt_passthrough` 是否符合 no-mixer / 直接 prompt 路线的用途
- 多 artist 下 `prompt`、`balanced`、`drift_auto` 是否是不同可控路线
- 新语法是否真实可执行
