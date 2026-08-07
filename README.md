# SitGuard

SitGuard 是一个**久坐姿态守护**桌面应用：通过摄像头检测是否有人就座、坐姿是否端正，并在久坐或姿态不良时弹出系统提醒。

本仓库包含两部分：

## 1. `sitguard-rs/` — Rust / Tauri 重写（主版本，推荐）

用 **Tauri 2 + Vue 3 + tract-onnx** 从零重写，纯 Rust 推理、单文件可执行、无 OpenCV 依赖。

- **M1** 骨架：Tauri + Vue 3 工程、摄像头采集（`nokhwa`）、预览事件流。
- **M2** 人脸检测：YuNet ONNX（`tract-onnx` 推理，避开 onnxruntime 二进制下载）。
- **M3** 坐姿判定：在场/专注状态机 + 五信号融合姿态评分。
- **M4** 久坐提醒：计时、系统托盘、通知、设置持久化、报告页。
- **M5** 全身姿态：MoveNet 17 关键点骨架叠加（模型缺失时优雅降级）。
- **M6** 躯干角度分析：骨架接入判定（肩线倾角 / 脊柱侧弯 / 头下沉比 / 躯干压缩比），
  与人脸几何融合；姿态标签加 hysteresis、连续量走 EMA，消除逐帧抖动。

### 坐姿是怎么判的

判定同时用两路信号，取更差的一路作为结论：

- **人脸几何**（YuNet）：脸在画面中的位置漂移、脸的大小（≈离屏距离）、yaw/pitch 头部朝向。
- **躯干角度**（MoveNet 17 关键点）：由鼻 / 双肩 / 双髋算出肩线倾角、脊柱侧弯角、
  头下沉比、躯干压缩比，关键点置信度 >0.3 才采信。

两路都是**相对你自己的基线**判断，不是绝对标准——基线在"专注且稳定"时自动快照，
也可以在界面上手动重新标定。若一开始就坐姿不良，基线会记住这个姿势，建议手动标定一次。

骨架缺失或置信度不足时，躯干分支保持中立，退化为**纯人脸几何**判定（前端会显示
"降级(仅人脸)"徽标），行为与 v0.1.0 一致。

构建与运行：

```bash
cd sitguard-rs
npm install
npm run tauri dev      # 开发预览
npm run tauri build    # 产出便携 exe（target/release/sitguard-rs.exe）
```

> 模型文件：`src-tauri/models/`（YuNet 内嵌，MoveNet 运行时加载）。
> 摄像头后端坑（MSMF 黑屏）已在 `nokhwa` 采集层用暖帧 + 黑帧校验规避。

### tract 本地补丁（`src-tauri/vendor/`）

跑通 MoveNet 需要修 tract 0.21.17 的三个上游 bug，补丁以 vendored crate 形式随仓库分发，
由 `src-tauri/Cargo.toml` 的 `[patch.crates-io]` 接入；每个被改的文件顶部都写清了
bug 表现与修法。**只有第一个会崩溃，另外两个是静默的数值错误**——模型照常加载、
输出形状照常是 `[1,1,17,3]`，但数值不对，所以单靠"能跑"无法发现。

| crate | 文件 | bug |
|---|---|---|
| `tract-hir` | `ops/array/gather_nd.rs` | GatherND 形状推导用了 indices 的 rank/shape 而非 data 的，`indices_rank - n` 在 usize 上下溢成 1.8e19，规则求解器无限 push 直到 ~2 GiB 分配失败、进程 abort（0.23.4 仍未修） |
| `tract-onnx` | `ops/resize.rs` | Resize 拿到**空 `scales`** 输入（tf2onnx 复用了 `roi` 的空 initializer）时仍走"有 scales"分支，结果一次上采样都不做，张量原样返回 |
| `tract-onnx` | `ops/resize.rs` | `half_pixel` 在左/上边界**外推**而非 clamp：`x_in = -0.25` 时整数索引被夹到 0，但 `x_frac` 仍是 -0.25，插值退化成 `1.25*y_left - 0.25*y_right`，让 ReLU 后的非负特征图出现负值，进而翻转关键点解码里的 ArgMax |

`pose::tests::movenet_matches_onnxruntime_reference` 用 onnxruntime 跑出的 17 个关键点
golden 值锁住这三处修复，防止升级 tract 时静默回归。

## CI 与发布

- **CI**（`.github/workflows/ci.yml`）：每次 push/PR 到 `main` 自动跑 `cargo fmt`/`clippy`、`cargo test --lib`（26 个单测，含 MoveNet/YuNet 真实推理与 ORT 数值对齐、骨架角度与标签 hysteresis）、前端类型检查 + 构建。
- **Release**（`.github/workflows/release.yml`）：打 `v*` 标签自动构建便携 exe，并把 exe + ONNX 模型作为 Release 资产上传（用 `bundle.targets: []` 跳过 GitHub 被墙的 WiX 安装包下载）。

```bash
git tag v0.2.0 && git push origin v0.2.0   # 触发 Release 构建
```

## 2. `recovered_reference/` — Python 原版参考实现

从丢失的源码中恢复的**可读参考**（仅算法对照用，不可直接运行）：

- `detector.py` — YuNet 人脸检测 + 坐姿判定原始实现。
- `cli.py` / `cli_gui.py` — 原始命令行 / GUI 入口。
- `__main__.py` — 入口。

## 3. `docs/`

- `CAMERA_TROUBLESHOOTING.md` — Windows 摄像头黑屏排障（MSMF vs DSHOW 后端坑）。
- `plans/` — 设计规划文档。

---

## 分支说明

- `main` — 最新 Rust 重写（含 M1–M5）。
- `master` / `v4-speedup` / `recovery-baseline` / `opencv-5-upgrade` — 历史阶段分支。

## 许可证

见 `LICENSE`。
