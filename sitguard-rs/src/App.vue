<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

interface CameraInfo {
  index: number;
  name: string;
  description: string;
}

interface ProbeResult {
  index: number;
  opened: boolean;
  width: number;
  height: number;
  mean_brightness: number;
  black_frame: boolean;
  message: string;
}

interface FaceDet {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  landmarks: number[][];
  yaw_deg: number;
  pitch_deg: number;
}

interface FramePayload {
  jpeg_base64: string;
  width: number;
  height: number;
  mean_brightness: number;
  seq: number;
  faces: FaceDet[];
  detect_ms: number;
  posture: PostureInfo;
  body: number[][] | null; // 17 × [x_px, y_px, score]
}

type PresenceState = "not_present" | "present_idle" | "present_focused";
type PostureState =
  | "unknown"
  | "good"
  | "head_down"
  | "slouching"
  | "lean_back"
  | "leaning";

interface PostureInfo {
  presence: PresenceState;
  posture: PostureState;
  posture_score: number;
  baseline_calibrated: boolean;
  focused_since: number;
}

interface Settings {
  baseline: Record<string, number>;
  thresholds: Record<string, number>;
  reminder_title: string;
  reminder_body: string;
  posture_reminder_title: string;
  posture_reminder_body: string;
}

interface SessionStats {
  sitting_seconds: number;
  focus_seconds: number;
  break_reminders: number;
  posture_reminders: number;
  next_reminder_minutes: number;
  secs_unknown: number;
  secs_good: number;
  secs_head_down: number;
  secs_slouching: number;
  secs_lean_back: number;
  secs_leaning: number;
}

// ---- UI state ----
const tab = ref<"live" | "report" | "settings">("live");
const cameras = ref<CameraInfo[]>([]);
const selected = ref<number>(0);
const probing = ref(false);
const probe = ref<ProbeResult | null>(null);
const previewing = ref(false);
const hasFrame = ref(false);
const brightness = ref<number>(0);
const fps = ref<number>(0);
const faceCount = ref<number>(0);
const detectMs = ref<number>(0);
const headPose = ref<string>("");
const bodyPresent = ref<boolean>(false);
const statusMsg = ref<string>("正在枚举摄像头…");
const canvasEl = ref<HTMLCanvasElement | null>(null);

const presence = ref<PresenceState>("not_present");
const postureState = ref<PostureState>("unknown");
const postureScore = ref<number>(0);
const baselineCalibrated = ref<boolean>(false);

const presenceLabel: Record<PresenceState, string> = {
  not_present: "不在位",
  present_idle: "在位 · 分心",
  present_focused: "在位 · 专注",
};
const postureLabel: Record<PostureState, string> = {
  unknown: "未知",
  good: "良好",
  head_down: "低头",
  slouching: "前倾",
  lean_back: "后仰",
  leaning: "偏移",
};

// ---- Settings / report state ----
const form = ref<Settings | null>(null);
const stats = ref<SessionStats | null>(null);

const presencePillClass = (p: PresenceState) => "p-" + p;
const posturePillClass = (p: PostureState) => "s-" + p;

let unlisten: UnlistenFn | null = null;
let trayUnlisten: UnlistenFn | null = null;
let frameCount = 0;
let fpsTimer: number | undefined;
let statsTimer: number | undefined;

const LM_COLORS = ["#22d3ee", "#22d3ee", "#f59e0b", "#a78bfa", "#a78bfa"]; // eyes, nose, mouth

// COCO 17-keypoint skeleton bone connections (index pairs).
const SKELETON: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [7, 9],
  [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13],
  [13, 15], [12, 14], [14, 16],
];
const KP_THRESHOLD = 0.3;

function drawFrame(p: FramePayload) {
  const canvas = canvasEl.value;
  if (!canvas) return;
  const img = new Image();
  img.onload = () => {
    if (canvas.width !== p.width || canvas.height !== p.height) {
      canvas.width = p.width;
      canvas.height = p.height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(img, 0, 0);

    const lw = Math.max(2, p.width / 480);

    // Body skeleton (MoveNet 17 keypoints), drawn under the face boxes.
    if (p.body) {
      const k = p.body;
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = lw;
      for (const [a, b] of SKELETON) {
        const ka = k[a];
        const kb = k[b];
        if (ka && kb && ka[2] > KP_THRESHOLD && kb[2] > KP_THRESHOLD) {
          ctx.beginPath();
          ctx.moveTo(ka[0], ka[1]);
          ctx.lineTo(kb[0], kb[1]);
          ctx.stroke();
        }
      }
      ctx.fillStyle = "#f43f5e";
      for (const kp of k) {
        if (kp[2] > KP_THRESHOLD) {
          ctx.beginPath();
          ctx.arc(kp[0], kp[1], lw * 1.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    for (const f of p.faces) {
      ctx.strokeStyle = "#4ade80";
      ctx.lineWidth = lw;
      ctx.strokeRect(f.x, f.y, f.w, f.h);
      const label = `${(f.score * 100).toFixed(0)}%  yaw ${f.yaw_deg.toFixed(0)}°  pitch ${f.pitch_deg.toFixed(0)}°`;
      ctx.font = `${Math.max(12, p.width / 55)}px system-ui`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(f.x, Math.max(0, f.y - 22), tw + 10, 20);
      ctx.fillStyle = "#4ade80";
      ctx.fillText(label, f.x + 5, Math.max(14, f.y - 7));
      f.landmarks.forEach((lm, i) => {
        ctx.fillStyle = LM_COLORS[i] ?? "#fff";
        ctx.beginPath();
        ctx.arc(lm[0], lm[1], lw * 1.6, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  };
  img.src = `data:image/jpeg;base64,${p.jpeg_base64}`;
}

function fmtDuration(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}小时${m}分`;
  if (m > 0) return `${m}分${r}秒`;
  return `${r}秒`;
}

const postureBars = [
  { key: "secs_good", label: "良好", cls: "s-good" },
  { key: "secs_head_down", label: "低头", cls: "s-head_down" },
  { key: "secs_slouching", label: "前倾", cls: "s-slouching" },
  { key: "secs_lean_back", label: "后仰", cls: "s-lean_back" },
  { key: "secs_leaning", label: "偏移", cls: "s-leaning" },
  { key: "secs_unknown", label: "未知", cls: "s-unknown" },
] as const;

function posturePct(key: string): number {
  if (!stats.value) return 0;
  const total =
    stats.value.secs_unknown +
    stats.value.secs_good +
    stats.value.secs_head_down +
    stats.value.secs_slouching +
    stats.value.secs_lean_back +
    stats.value.secs_leaning;
  if (total <= 0) return 0;
  return (Math.max(0, (stats.value as any)[key] as number) / total) * 100;
}

async function refreshCameras() {
  statusMsg.value = "正在枚举摄像头…";
  cameras.value = await invoke<CameraInfo[]>("list_cameras");
  if (cameras.value.length === 0) {
    statusMsg.value = "未检测到摄像头";
  } else {
    selected.value = cameras.value[0].index;
    statusMsg.value = `检测到 ${cameras.value.length} 个摄像头`;
  }
}

async function doProbe() {
  probing.value = true;
  probe.value = null;
  statusMsg.value = "正在探测（暖帧 + 黑帧校验）…";
  try {
    probe.value = await invoke<ProbeResult>("probe_camera", { index: selected.value });
    statusMsg.value = probe.value.black_frame
      ? "⚠️ 摄像头打开成功但输出全黑帧（疑似固件/隐私开关）"
      : `探测通过：${probe.value.width}x${probe.value.height} 亮度 ${probe.value.mean_brightness.toFixed(1)}`;
  } catch (e) {
    statusMsg.value = `探测失败：${e}`;
  } finally {
    probing.value = false;
  }
}

async function startPreview() {
  if (previewing.value) return;
  frameCount = 0;
  unlisten = await listen<FramePayload>("camera-frame", (ev) => {
    hasFrame.value = true;
    brightness.value = ev.payload.mean_brightness;
    faceCount.value = ev.payload.faces.length;
    detectMs.value = ev.payload.detect_ms;
    headPose.value =
      ev.payload.faces.length > 0
        ? `yaw ${ev.payload.faces[0].yaw_deg.toFixed(0)}° / pitch ${ev.payload.faces[0].pitch_deg.toFixed(0)}°`
        : "";
    presence.value = ev.payload.posture.presence;
    postureState.value = ev.payload.posture.posture;
    postureScore.value = ev.payload.posture.posture_score;
    baselineCalibrated.value = ev.payload.posture.baseline_calibrated;
    bodyPresent.value = ev.payload.body != null;
    drawFrame(ev.payload);
    frameCount++;
  });
  fpsTimer = window.setInterval(() => {
    fps.value = frameCount;
    frameCount = 0;
  }, 1000);
  try {
    await invoke("start_preview", { index: selected.value });
    previewing.value = true;
    statusMsg.value = "预览中";
  } catch (e) {
    statusMsg.value = `启动预览失败：${e}`;
    unlisten?.();
    unlisten = null;
    window.clearInterval(fpsTimer);
  }
}

async function stopPreview() {
  await invoke("stop_preview");
  previewing.value = false;
  unlisten?.();
  unlisten = null;
  window.clearInterval(fpsTimer);
  statusMsg.value = "预览已停止";
}

async function recalibrate() {
  try {
    await invoke("calibrate_baseline");
    statusMsg.value = "已请求重新标定基线（下一帧有人脸即生效）";
  } catch (e) {
    statusMsg.value = `标定请求失败：${e}`;
  }
}

async function pauseReminders() {
  try {
    await invoke("pause_session", { minutes: 10 });
    statusMsg.value = "已暂缓提醒 10 分钟";
  } catch (e) {
    statusMsg.value = `暂缓失败：${e}`;
  }
}

async function loadSettings() {
  try {
    form.value = await invoke<Settings>("get_settings");
  } catch (e) {
    statusMsg.value = `加载设置失败：${e}`;
  }
}

async function saveSettings() {
  if (!form.value) return;
  try {
    await invoke("update_settings", { new: form.value });
    statusMsg.value = "设置已保存";
  } catch (e) {
    statusMsg.value = `保存设置失败：${e}`;
  }
}

async function refreshStats() {
  try {
    stats.value = await invoke<SessionStats>("get_session_stats");
  } catch {
    /* not running yet */
  }
}

onMounted(async () => {
  refreshCameras();
  loadSettings();
  refreshStats();
  statsTimer = window.setInterval(refreshStats, 2000);
  trayUnlisten = await listen<string>("tray-command", (ev) => {
    if (ev.payload === "start" && !previewing.value) startPreview();
    if (ev.payload === "stop" && previewing.value) stopPreview();
  });
});

onUnmounted(() => {
  if (previewing.value) stopPreview();
  trayUnlisten?.();
  window.clearInterval(statsTimer);
});
</script>

<template>
  <main class="container">
    <h1>SitGuard <span class="badge">Rust</span></h1>

    <div class="tabs">
      <button :class="{ active: tab === 'live' }" @click="tab = 'live'">实时</button>
      <button :class="{ active: tab === 'report' }" @click="tab = 'report'">报告</button>
      <button :class="{ active: tab === 'settings' }" @click="tab = 'settings'">设置</button>
    </div>

    <!-- LIVE -->
    <section v-if="tab === 'live'">
      <div class="toolbar">
        <select v-model.number="selected" :disabled="previewing">
          <option v-for="c in cameras" :key="c.index" :value="c.index">
            [{{ c.index }}] {{ c.name }}
          </option>
        </select>
        <button @click="refreshCameras" :disabled="previewing">刷新</button>
        <button @click="doProbe" :disabled="probing || previewing">探测</button>
        <button v-if="!previewing" class="primary" @click="startPreview">开始预览</button>
        <button v-else class="danger" @click="stopPreview">停止预览</button>
      </div>

      <p class="status">{{ statusMsg }}</p>

      <div class="preview">
        <canvas ref="canvasEl" v-show="hasFrame"></canvas>
        <div v-if="!hasFrame" class="placeholder">无画面</div>
        <div v-if="previewing" class="overlay">
          亮度 {{ brightness.toFixed(1) }} · {{ fps }} fps · 推理 {{ detectMs.toFixed(1) }}ms
          <span v-if="faceCount > 0"> · 人脸 {{ faceCount }}</span>
          <span v-else> · 未检测到人脸</span>
          <span v-if="bodyPresent"> · 骨架✓</span>
        </div>
      </div>

      <div v-if="previewing" class="posture-panel">
        <div class="pill" :class="presencePillClass(presence)">
          <span class="k">在场</span>{{ presenceLabel[presence] }}
        </div>
        <div class="pill" :class="posturePillClass(postureState)">
          <span class="k">坐姿</span>{{ postureLabel[postureState] }}
        </div>
        <div class="pill">
          <span class="k">评分</span>{{ postureScore.toFixed(0) }}
        </div>
        <div class="pill" :class="baselineCalibrated ? 'ok' : 'warn'">
          <span class="k">基线</span>{{ baselineCalibrated ? "已标定" : "标定中…" }}
        </div>
        <button class="ghost" @click="recalibrate">重新标定</button>
        <button class="ghost" @click="pauseReminders">暂缓提醒 10 分钟</button>
      </div>

      <div v-if="probe" class="probe-card" :class="{ warn: probe.black_frame }">
        <div>打开：{{ probe.opened ? "成功" : "失败" }}</div>
        <div>分辨率：{{ probe.width }}x{{ probe.height }}</div>
        <div>平均亮度：{{ probe.mean_brightness.toFixed(1) }}</div>
        <div>黑帧：{{ probe.black_frame ? "是 ⚠️" : "否" }}</div>
        <div>{{ probe.message }}</div>
      </div>
    </section>

    <!-- REPORT -->
    <section v-else-if="tab === 'report'">
      <h2>本次会话统计</h2>
      <div v-if="stats" class="report-grid">
        <div class="report-card">
          <div class="big">{{ fmtDuration(stats.sitting_seconds) }}</div>
          <div class="cap">累计在座时长</div>
        </div>
        <div class="report-card">
          <div class="big">{{ fmtDuration(stats.focus_seconds) }}</div>
          <div class="cap">专注时长</div>
        </div>
        <div class="report-card">
          <div class="big">{{ stats.break_reminders }}</div>
          <div class="cap">休息提醒次数</div>
        </div>
        <div class="report-card">
          <div class="big">{{ stats.posture_reminders }}</div>
          <div class="cap">姿态提醒次数</div>
        </div>
        <div class="report-card">
          <div class="big">{{ stats.next_reminder_minutes.toFixed(0) }}′</div>
          <div class="cap">距下次休息提醒</div>
        </div>
      </div>

      <h3>坐姿分布</h3>
      <div class="bars">
        <div v-for="b in postureBars" :key="b.key" class="bar-row">
          <span class="bar-label">{{ b.label }}</span>
          <div class="bar-track">
            <div class="bar-fill" :class="b.cls" :style="{ width: posturePct(b.key) + '%' }"></div>
          </div>
          <span class="bar-pct">{{ posturePct(b.key).toFixed(0) }}%</span>
        </div>
      </div>
      <p class="hint">统计在“开始预览”后实时累计，停止预览时自动保存已标定的基线。</p>
    </section>

    <!-- SETTINGS -->
    <section v-else-if="tab === 'settings'">
      <h2>设置</h2>
      <div v-if="form" class="settings-form">
        <label>
          连续坐姿提醒阈值（分钟）
          <input type="number" min="1" step="1" v-model.number="form.thresholds.sitting_threshold_minutes" />
        </label>
        <label>
          建议休息时长（分钟）
          <input type="number" min="1" step="1" v-model.number="form.thresholds.break_minutes" />
        </label>
        <label>
          休息提醒冷却（分钟）
          <input type="number" min="1" step="1" v-model.number="form.thresholds.reminder_cooldown_minutes" />
        </label>
        <label>
          不良姿态持续触发（秒）
          <input type="number" min="1" step="1" v-model.number="form.thresholds.posture_dwell_seconds" />
        </label>
        <label>
          姿态提醒冷却（分钟）
          <input type="number" min="1" step="1" v-model.number="form.thresholds.posture_cooldown_minutes" />
        </label>
        <label>
          休息提醒标题
          <input type="text" v-model="form.reminder_title" />
        </label>
        <label>
          休息提醒内容（支持 {minutes} / {break}）
          <input type="text" v-model="form.reminder_body" />
        </label>
        <label>
          姿态提醒标题
          <input type="text" v-model="form.posture_reminder_title" />
        </label>
        <label>
          姿态提醒内容
          <input type="text" v-model="form.posture_reminder_body" />
        </label>
        <div class="form-actions">
          <button class="primary" @click="saveSettings">保存设置</button>
          <button class="ghost" @click="loadSettings">重置为已保存</button>
        </div>
      </div>
      <p v-else class="hint">正在加载设置…</p>
    </section>
  </main>
</template>

<style scoped>
.container {
  max-width: 860px;
  margin: 0 auto;
  padding: 16px;
  font-family: "Segoe UI", system-ui, sans-serif;
}
h1 {
  font-size: 22px;
  margin: 0 0 12px;
}
h2 {
  font-size: 17px;
  margin: 18px 0 10px;
}
h3 {
  font-size: 15px;
  margin: 16px 0 8px;
}
.badge {
  font-size: 11px;
  background: #ce422b;
  color: #fff;
  border-radius: 4px;
  padding: 2px 6px;
  vertical-align: middle;
}
.tabs {
  display: flex;
  gap: 6px;
  margin-bottom: 12px;
}
.tabs button {
  padding: 6px 16px;
  border-radius: 6px 6px 0 0;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  cursor: pointer;
  color: #475569;
}
.tabs button.active {
  background: #2563eb;
  color: #fff;
  border-color: #2563eb;
}
.toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
select,
button {
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid #ccc;
  background: #fff;
  cursor: pointer;
}
button.primary {
  background: #2563eb;
  color: #fff;
  border-color: #2563eb;
}
button.danger {
  background: #dc2626;
  color: #fff;
  border-color: #dc2626;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.status {
  color: #666;
  font-size: 13px;
  min-height: 18px;
}
.preview {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #111;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}
.preview canvas {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.placeholder {
  color: #555;
}
.overlay {
  position: absolute;
  right: 8px;
  bottom: 8px;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  font-size: 12px;
  padding: 3px 8px;
  border-radius: 4px;
}
.probe-card {
  margin-top: 12px;
  padding: 10px 14px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 13px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 4px;
}
.probe-card.warn {
  border-color: #f59e0b;
  background: #fffbeb;
}
.posture-panel {
  margin-top: 12px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 13px;
  background: #f1f5f9;
  color: #0f172a;
  border: 1px solid #e2e8f0;
}
.pill .k {
  color: #64748b;
  font-size: 11px;
}
.pill.ok {
  background: #ecfdf5;
  border-color: #6ee7b7;
}
.pill.warn {
  background: #fffbeb;
  border-color: #fcd34d;
}
.p-present_focused {
  background: #ecfdf5;
  border-color: #6ee7b7;
}
.p-present_idle {
  background: #eff6ff;
  border-color: #93c5fd;
}
.p-not_present {
  background: #f8fafc;
  border-color: #cbd5e1;
}
.s-good {
  background: #ecfdf5;
  border-color: #6ee7b7;
}
.s-head_down {
  background: #fef2f2;
  border-color: #fca5a5;
}
.s-slouching {
  background: #fff7ed;
  border-color: #fdba74;
}
.s-lean_back {
  background: #f0f9ff;
  border-color: #7dd3fc;
}
.s-leaning {
  background: #fefce8;
  border-color: #fde047;
}
.s-unknown {
  background: #f8fafc;
  border-color: #cbd5e1;
}
button.ghost {
  background: #fff;
  border: 1px solid #cbd5e1;
  color: #334155;
  padding: 5px 11px;
  border-radius: 999px;
  cursor: pointer;
  font-size: 13px;
}
button.ghost:hover {
  background: #f8fafc;
}
/* report */
.report-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
}
.report-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 14px;
  text-align: center;
}
.report-card .big {
  font-size: 22px;
  font-weight: 600;
  color: #0f172a;
}
.report-card .cap {
  font-size: 12px;
  color: #64748b;
  margin-top: 4px;
}
.bars {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.bar-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.bar-label {
  width: 48px;
  font-size: 13px;
  color: #334155;
}
.bar-track {
  flex: 1;
  height: 14px;
  background: #f1f5f9;
  border-radius: 7px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 7px;
}
.bar-pct {
  width: 42px;
  text-align: right;
  font-size: 12px;
  color: #64748b;
}
.hint {
  color: #94a3b8;
  font-size: 12px;
  margin-top: 10px;
}
/* settings */
.settings-form {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.settings-form label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  color: #334155;
}
.settings-form input {
  padding: 7px 9px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 14px;
}
.form-actions {
  grid-column: 1 / -1;
  display: flex;
  gap: 10px;
  margin-top: 4px;
}
</style>
