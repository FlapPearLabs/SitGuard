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

构建与运行：

```bash
cd sitguard-rs
npm install
npm run tauri dev      # 开发预览
npm run tauri build    # 产出便携 exe（target/release/sitguard-rs.exe）
```

> 模型文件：`src-tauri/models/`（YuNet 内嵌，MoveNet 运行时加载）。
> 摄像头后端坑（MSMF 黑屏）已在 `nokhwa` 采集层用暖帧 + 黑帧校验规避。

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
