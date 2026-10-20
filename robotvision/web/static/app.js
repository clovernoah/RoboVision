/* RoboVision 前端逻辑：导航、图像分析、DQN 决策仿真 */
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const PAGE_META = {
    overview: ["系统概览", "行人 / 车辆 / 人脸检测 · 单目深度估计 · 深度强化学习决策"],
    detect: ["目标检测", "YOLOv8n 识别行人、车辆等 COCO 目标，支持视频流与 IoU 跟踪"],
    face: ["人脸检测", "YuNet 输出人脸框、置信度与 5 个关键点"],
    depth: ["视觉深度估计", "MiDaS 单目相对深度 + 检测融合，输出可解释的导航风险"],
    decision: ["机器人决策", "Double DQN 在动态栅格环境中学习避障与导航"],
  };

  const state = {
    samples: [],
    health: null,
    analyzers: {},
  };

  // ------------------------------------------------------------------
  // 基础 UI
  // ------------------------------------------------------------------
  let toastTimer = null;
  function toast(message, isError = false) {
    const el = $("#toast");
    el.textContent = message;
    el.classList.toggle("error", isError);
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 4200);
  }

  function switchView(name, updateHash = true) {
    if (!PAGE_META[name]) return;
    $$(".view").forEach((el) => el.classList.toggle("active", el.id === `view-${name}`));
    $$(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
    $("#page-title").textContent = PAGE_META[name][0];
    $("#page-sub").textContent = PAGE_META[name][1];
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (updateHash && location.hash !== `#${name}`) {
      history.replaceState(null, "", `${location.pathname}${location.search}#${name}`);
    }
    if (name === "decision") {
      decision.onShow();
    }
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
  }

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      throw new Error(`服务返回异常（HTTP ${response.status}）`);
    }
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  function metricHtml(items) {
    return items.map((item) => `
      <div class="metric">
        <span>${escapeHtml(item.label)}</span>
        <b class="${item.tone || ""}">${escapeHtml(item.value)}</b>
      </div>`).join("");
  }

  // ------------------------------------------------------------------
  // 健康状态与示例
  // ------------------------------------------------------------------
  function setStatusCell(id, ok, okText = "就绪", badText = "缺失") {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = ok ? okText : badText;
    el.className = ok ? "ok" : "bad";
  }

  async function loadHealth() {
    try {
      const data = await api("/api/health");
      state.health = data;
      state.samples = data.samples || [];
      const models = data.models || {};
      const ready = ["detection", "face", "depth"].every((key) => models[key] && models[key].exists);
      $("#health-dot").className = `dot ${ready ? "ok" : "bad"}`;
      $("#health-text").textContent = ready ? "感知模型就绪" : "部分模型缺失";
      setStatusCell("model-yolo", models.detection?.exists);
      setStatusCell("model-face", models.face?.exists);
      setStatusCell("model-depth", models.depth?.exists);
      $("#model-device").textContent = models.device || "cpu";
      const ckpts = data.checkpoints || {};
      setStatusCell("ckpt-robotnav", ckpts.robotnav?.exists, "已训练", "未训练");
      setStatusCell("ckpt-cartpole", ckpts.cartpole?.exists, "已训练", "未训练");
      if (!ready) {
        toast("检测到模型文件缺失，请运行 scripts/download_models.py", true);
      }
      renderOverviewSamples();
      Object.values(state.analyzers).forEach((analyzer) => analyzer.renderSamples());
      maybeAutoDemo();
    } catch (error) {
      $("#health-dot").className = "dot bad";
      $("#health-text").textContent = "服务不可用";
      toast(`无法连接后端：${error.message}`, true);
    }
  }

  function maybeAutoDemo() {
    const params = new URLSearchParams(location.search);
    const demo = params.get("demo");
    if (!demo) return;
    if (demo === "decision") {
      switchView("decision");
      setTimeout(() => decision.reset(), 300);
      return;
    }
    if (!PAGE_META[demo] || !state.analyzers[demo]) return;
    const defaults = { detect: "street_traffic.jpg", face: "group.jpg", depth: "city_street.jpg" };
    const sample = params.get("sample") || defaults[demo];
    switchView(demo);
    setTimeout(() => {
      state.analyzers[demo].loadSample(sample).catch((error) => toast(error.message, true));
    }, 250);
  }

  function sampleTask(filename) {
    const sample = state.samples.find((item) => item.filename === filename);
    const scene = sample ? sample.scene : "";
    if (scene.includes("人脸")) return "face";
    if (scene.includes("深度")) return "depth";
    return "detect";
  }

  function renderOverviewSamples() {
    const container = $("#overview-samples");
    if (!state.samples.length) {
      container.innerHTML = '<div class="loading-line">尚未下载示例图片，请运行 python scripts/download_assets.py</div>';
      return;
    }
    container.innerHTML = state.samples.map((sample) => `
      <button class="sample-card" data-sample="${escapeHtml(sample.filename)}" title="${escapeHtml(sample.scene)}">
        <img src="${sample.url}" loading="lazy" alt="${escapeHtml(sample.scene)}">
        <span>${escapeHtml(sample.scene)}</span>
      </button>`).join("");
    $$(".sample-card", container).forEach((card) => {
      card.addEventListener("click", () => {
        const filename = card.dataset.sample;
        const task = sampleTask(filename);
        switchView(task);
        const analyzer = state.analyzers[task];
        if (analyzer) analyzer.loadSample(filename).catch((error) => toast(error.message, true));
      });
    });
  }

  // ------------------------------------------------------------------
  // 图像分析
  // ------------------------------------------------------------------
  class Analyzer {
    constructor(task, tasksParam) {
      this.task = task;
      this.tasksParam = tasksParam;
      this.root = document.getElementById(`view-${task}`);
      this.dropzone = $(".dropzone", this.root);
      this.input = $("input[type=file]", this.dropzone);
      this.previewWrap = $(`#preview-${task}`);
      this.previewImg = $("img", this.previewWrap);
      this.runBtn = $(`.run-btn[data-task="${task}"]`, this.root);
      this.media = $(`#media-${task}`);
      this.metrics = $(`#metrics-${task}`);
      this.table = $(`#table-${task}`);
      this.latency = $(`#latency-${task}`);
      this.sampleRow = $(`#samples-${task}`);
      this.file = null;
      this.previewUrl = null;
      this.busy = false;
      this.bind();
    }

    bind() {
      this.dropzone.addEventListener("click", () => this.input.click());
      this.input.addEventListener("change", () => {
        if (this.input.files && this.input.files[0]) this.setFile(this.input.files[0]);
      });
      ["dragenter", "dragover"].forEach((name) => {
        this.dropzone.addEventListener(name, (event) => {
          event.preventDefault();
          this.dropzone.classList.add("drag");
        });
      });
      ["dragleave", "drop"].forEach((name) => {
        this.dropzone.addEventListener(name, (event) => {
          event.preventDefault();
          this.dropzone.classList.remove("drag");
        });
      });
      this.dropzone.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) this.setFile(file);
      });
      this.runBtn.addEventListener("click", () => this.run());
    }

    setFile(file) {
      if (!file.type.startsWith("image/")) {
        toast("请选择图片文件", true);
        return;
      }
      this.file = file;
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = URL.createObjectURL(file);
      this.previewImg.src = this.previewUrl;
      this.previewWrap.classList.add("show");
      this.runBtn.disabled = false;
    }

    renderSamples() {
      if (!this.sampleRow) return;
      const samples = state.samples.slice(0, 8);
      this.sampleRow.innerHTML = samples.map((sample) => `
        <button class="sample-thumb" data-sample="${escapeHtml(sample.filename)}" title="${escapeHtml(sample.scene)}">
          <img src="${sample.url}" loading="lazy" alt="">
          <span>${escapeHtml(sample.filename.replace(/\.[^.]+$/, ""))}</span>
        </button>`).join("");
      $$(".sample-thumb", this.sampleRow).forEach((thumb) => {
        thumb.addEventListener("click", () => {
          this.loadSample(thumb.dataset.sample).catch((error) => toast(error.message, true));
        });
      });
    }

    async loadSample(filename) {
      const sample = state.samples.find((item) => item.filename === filename);
      if (!sample) throw new Error("示例图片不存在");
      const response = await fetch(sample.url);
      if (!response.ok) throw new Error("示例图片加载失败");
      const blob = await response.blob();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      this.setFile(file);
      await this.run();
    }

    async run() {
      if (!this.file) {
        toast("请先选择一张图片", true);
        return;
      }
      if (this.busy) return;
      this.busy = true;
      this.runBtn.disabled = true;
      const original = this.runBtn.textContent;
      this.runBtn.textContent = "推理中…";
      this.media.innerHTML = '<div class="empty">正在执行神经网络推理…</div>';
      this.metrics.innerHTML = "";
      this.table.innerHTML = "";
      this.latency.textContent = "推理中";

      const form = new FormData();
      form.append("file", this.file);
      form.append("tasks", this.tasksParam);
      try {
        const data = await api("/api/analyze", { method: "POST", body: form });
        this.render(data);
      } catch (error) {
        console.error("[analyzer]", error);
        this.media.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
        this.latency.textContent = "失败";
        toast(error.message, true);
      } finally {
        this.busy = false;
        this.runBtn.disabled = false;
        this.runBtn.textContent = original;
      }
    }

    render(data) {
      if (this.task === "detect") this.renderDetection(data);
      else if (this.task === "face") this.renderFace(data);
      else this.renderDepth(data);
    }

    renderDetection(data) {
      const count = data.counts || {};
      const timings = data.timings_ms || {};
      this.latency.textContent = `检测 ${timings.detect ?? "—"} ms · 总计 ${timings.total ?? "—"} ms`;
      this.media.innerHTML = `<img src="${data.images.annotated}" alt="检测结果">`;
      this.metrics.innerHTML = metricHtml([
        { label: "目标总数", value: count.objects ?? 0, tone: "mint" },
        { label: "行人", value: count.person ?? 0, tone: "cyan" },
        { label: "车辆", value: count.vehicle ?? 0, tone: "amber" },
        { label: "推理耗时", value: `${timings.detect ?? "—"} ms` },
      ]);
      this.table.innerHTML = this.detectionTable(data.detections, false);
    }

    renderFace(data) {
      const timings = data.timings_ms || {};
      const faces = data.faces || [];
      const image = data.images.faces || data.images.annotated;
      this.latency.textContent = `人脸 ${timings.face ?? "—"} ms · 总计 ${timings.total ?? "—"} ms`;
      this.media.innerHTML = `<img src="${image}" alt="人脸检测结果">`;
      const best = faces.length ? Math.max(...faces.map((face) => face.score)) : 0;
      this.metrics.innerHTML = metricHtml([
        { label: "人脸数量", value: faces.length, tone: "mint" },
        { label: "最高置信度", value: faces.length ? best.toFixed(3) : "—", tone: "cyan" },
        { label: "推理耗时", value: `${timings.face ?? "—"} ms` },
      ]);
      if (!faces.length) {
        this.table.innerHTML = '<div class="empty small">未检测到人脸，可换一张更清晰的正面照片</div>';
        return;
      }
      const rows = faces.map((face, index) => `
        <tr>
          <td>${index + 1}</td>
          <td><span class="pill class">face</span></td>
          <td>${face.score.toFixed(4)}</td>
          <td class="mono">${face.xyxy.map((v) => Math.round(v)).join(", ")}</td>
          <td>${face.landmarks ? face.landmarks.length : 0}</td>
        </tr>`).join("");
      this.table.innerHTML = `<table><thead><tr>
        <th>#</th><th>类别</th><th>置信度</th><th>框 (x1,y1,x2,y2)</th><th>关键点</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
    }

    renderDepth(data) {
      const timings = data.timings_ms || {};
      const fusion = data.fusion;
      const objects = fusion ? fusion.objects || [] : [];
      this.latency.textContent = `深度 ${timings.depth ?? "—"} ms · 总计 ${timings.total ?? "—"} ms`;
      this.media.innerHTML = `
        <img src="${data.images.depth_panel || data.images.depth}" alt="深度估计">
        <span class="media-caption">左：原始图像 · 右：MiDaS 相对深度（越亮越近）</span>
        ${data.images.fused ? `
          <img class="fused-img" src="${data.images.fused}" alt="深度融合">
          <span class="media-caption">深度融合：检测框与近 / 中 / 远风险档位</span>` : ""}`;
      const stats = data.depth ? data.depth.stats : {};
      const near = objects.filter((item) => item.band === "near").length;
      const mid = objects.filter((item) => item.band === "mid").length;
      this.metrics.innerHTML = metricHtml([
        { label: "相对深度均值", value: stats.mean !== undefined ? stats.mean.toFixed(3) : "—", tone: "mint" },
        { label: "近距目标", value: near, tone: "amber" },
        { label: "中距目标", value: mid, tone: "cyan" },
        { label: "推理耗时", value: `${timings.depth ?? "—"} ms` },
      ]);

      const suggestion = fusion ? fusion.suggestion : null;
      const box = $("#suggestion-depth");
      if (suggestion) {
        box.classList.add("show");
        box.innerHTML = `导航建议：<b>${escapeHtml(suggestion.action_label)}</b>
          <span class="reason">${escapeHtml(suggestion.reason)}</span>`;
      } else {
        box.classList.remove("show");
      }

      const grid = fusion ? fusion.grid : null;
      const corridor = fusion ? fusion.corridor : null;
      const heat = grid ? `
        <div>
          <div class="heatmap">
            ${grid.flat().map((value) => `
              <div class="heat-cell" style="background:${heatColor(value)}">${value.toFixed(2)}</div>`).join("")}
          </div>
          <div class="heat-label">深度接近度网格（左下/右下为近处地面）</div>
        </div>` : "";
      const bars = corridor ? `
        <div class="corridor-list">
          ${corridorBar("左侧", corridor.left)}
          ${corridorBar("正前", corridor.center)}
          ${corridorBar("右侧", corridor.right)}
        </div>` : "";
      $("#fusion-depth").innerHTML = `${heat}${bars}`;

      if (!objects.length) {
        this.table.innerHTML = '<div class="empty small">未检测到可融合的目标，仅展示深度图</div>';
        return;
      }
      const rows = objects.map((item) => `
        <tr>
          <td>${item.track_id !== null ? `#${item.track_id}` : "—"}</td>
          <td><span class="pill class">${escapeHtml(item.class_name)}</span></td>
          <td>${item.score.toFixed(3)}</td>
          <td>${item.proximity.toFixed(3)}</td>
          <td><span class="pill ${item.band}">${escapeHtml(item.band_label)}</span></td>
          <td>${escapeHtml(item.direction_label)}</td>
          <td><span class="pill ${item.risk}">${riskLabel(item.risk)}</span></td>
        </tr>`).join("");
      this.table.innerHTML = `<table><thead><tr>
        <th>ID</th><th>类别</th><th>置信度</th><th>接近度</th><th>距离档</th><th>方位</th><th>风险</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
    }

    detectionTable(detections, showTrack = true) {
      if (!detections || !detections.length) {
        return '<div class="empty small">未检测到目标，可尝试降低阈值或更换图片</div>';
      }
      const rows = detections.map((item) => `
        <tr>
          ${showTrack ? `<td>${item.track_id !== null ? `#${item.track_id}` : "—"}</td>` : ""}
          <td><span class="pill class">${escapeHtml(item.class_name)}</span></td>
          <td>${item.score.toFixed(4)}</td>
          <td class="mono">${item.xyxy.map((v) => Math.round(v)).join(", ")}</td>
          <td class="mono">${Math.round(item.xyxy[2] - item.xyxy[0])} × ${Math.round(item.xyxy[3] - item.xyxy[1])}</td>
        </tr>`).join("");
      return `<table><thead><tr>
        ${showTrack ? "<th>ID</th>" : ""}<th>类别</th><th>置信度</th><th>框 (x1,y1,x2,y2)</th><th>尺寸</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
    }
  }

  function riskLabel(risk) {
    return { high: "高", medium: "中", low: "低" }[risk] || risk;
  }

  function heatColor(value) {
    const t = Math.max(0, Math.min(1, value));
    const lerp = (a, b, k) => Math.round(a + (b - a) * k);
    let r, g, b;
    if (t < 0.5) {
      const k = t / 0.5;
      r = lerp(52, 251, k); g = lerp(211, 191, k); b = lerp(153, 36, k);
    } else {
      const k = (t - 0.5) / 0.5;
      r = lerp(251, 248, k); g = lerp(191, 113, k); b = lerp(36, 113, k);
    }
    return `rgba(${r}, ${g}, ${b}, ${(0.22 + 0.7 * t).toFixed(2)})`;
  }

  function corridorBar(label, value) {
    const percent = Math.max(0, Math.min(1, value)) * 100;
    return `<div class="corridor-item">
      <span>${label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${percent}%"></div></div>
      <span class="mono">${value.toFixed(2)}</span>
    </div>`;
  }

  // ------------------------------------------------------------------
  // 决策仿真
  // ------------------------------------------------------------------
  const DECISION_META = {
    robotnav: { title: "机器人栅格导航", tag: "Double DQN", policies: [["dqn", "Double DQN"], ["astar", "A* 基线"], ["random", "随机策略"], ["manual", "手动控制"]] },
    cartpole: { title: "CartPole 平衡", tag: "Double DQN", policies: [["dqn", "Double DQN"], ["heuristic", "启发式"], ["random", "随机策略"], ["manual", "手动控制"]] },
  };
  const ACTION_LABELS = { robotnav: ["上", "下", "左", "右"], cartpole: ["左推", "右推"] };

  const decision = {
    env: "robotnav",
    policy: "dqn",
    auto: true,
    speed: 220,
    running: false,
    busy: false,
    started: false,
    done: false,
    state: null,
    envType: "robotnav",
    timer: null,

    init() {
      this.envSelect = $("#env-select");
      this.policySelect = $("#policy-select");
      this.autoRun = $("#auto-run");
      this.speedInput = $("#speed");
      this.overlay = $("#sim-overlay");
      this.canvas = $("#grid-canvas");
      this.ctx = this.canvas.getContext("2d");
      this.cartpoleImg = $("#cartpole-frame");

      this.updatePolicyOptions();
      this.envSelect.addEventListener("change", () => {
        this.stop();
        this.env = this.envSelect.value;
        this.updatePolicyOptions();
        this.updateMeta();
        this.started = false;
        this.setPlaceholder(`已切换到 ${DECISION_META[this.env].title}，点击「开始新回合」`);
        this.showEnvironment();
      });
      this.policySelect.addEventListener("change", () => {
        this.policy = this.policySelect.value;
        this.updateManualUI();
        if (this.started) this.reset();
      });
      $("#btn-reset").addEventListener("click", () => this.reset());
      $("#btn-step").addEventListener("click", () => {
        if (this.policy === "manual") {
          toast("手动模式请使用方向键控制机器人", true);
          return;
        }
        if (!this.started || this.done) return;
        this.stop();
        this.step(true);
      });
      this.autoRun.addEventListener("change", () => {
        this.auto = this.autoRun.checked;
        if (this.auto && this.policy !== "manual" && this.started && !this.done) this.startLoop();
        else this.stop();
      });
      this.speedInput.addEventListener("input", () => {
        this.speed = Number(this.speedInput.value);
        if (this.running) {
          this.stop();
          this.startLoop();
        }
      });
      document.addEventListener("keydown", (event) => this.onKey(event));
    },

    onShow() {
      if (!this.started) {
        this.showEnvironment();
        this.drawEmptyGrid();
      }
    },

    updatePolicyOptions() {
      const policies = DECISION_META[this.env].policies;
      this.policySelect.innerHTML = policies
        .map(([value, label]) => `<option value="${value}">${label}</option>`)
        .join("");
      this.policy = policies[0][0];
      this.policySelect.value = this.policy;
      this.updateManualUI();
      this.updateMeta();
    },

    updateMeta() {
      $("#sim-title").textContent = DECISION_META[this.env].title;
      $("#sim-tag").textContent = DECISION_META[this.env].tag;
    },

    updateManualUI() {
      const manual = this.policy === "manual";
      $("#manual-hint").textContent = manual ? "方向键控制机器人" : "自动策略可单步观察";
      this.autoRun.disabled = manual;
      if (manual) {
        this.auto = false;
        this.autoRun.checked = false;
      } else {
        this.auto = this.autoRun.checked;
      }
    },

    showEnvironment() {
      const nav = this.env === "robotnav";
      this.canvas.hidden = !nav;
      this.cartpoleImg.hidden = nav;
      if (nav) this.drawEmptyGrid();
      else this.cartpoleImg.src = "";
    },

    setPlaceholder(text) {
      this.overlay.innerHTML = `<span class="sim-message">${escapeHtml(text)}</span>`;
      this.overlay.classList.remove("hide");
    },

    async reset() {
      this.stop();
      this.busy = true;
      $("#btn-reset").disabled = true;
      this.setPlaceholder("正在创建环境…");
      try {
        const data = await api("/api/decision/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ env: this.env, policy: this.policy }),
        });
        this.started = true;
        this.done = false;
        this.applyPayload(data);
        $("#btn-step").disabled = false;
        this.overlay.classList.add("hide");
        const note = $("#decision-note");
        if (data.note) {
          note.textContent = data.note;
          note.hidden = false;
        } else {
          note.hidden = true;
        }
        if (this.auto && this.policy !== "manual") this.startLoop();
        else this.stop();
      } catch (error) {
        this.started = false;
        this.setPlaceholder("回合创建失败");
        toast(error.message, true);
      } finally {
        this.busy = false;
        $("#btn-reset").disabled = false;
      }
    },

    startLoop() {
      this.stop();
      if (this.policy === "manual") return;
      this.running = true;
      const tick = async () => {
        if (!this.running) return;
        if (!this.started || this.done) {
          this.stop();
          return;
        }
        await this.step(true);
        if (this.running && !this.done) {
          this.timer = setTimeout(tick, this.speed);
        }
      };
      this.timer = setTimeout(tick, this.speed);
    },

    stop() {
      this.running = false;
      if (this.timer) {
        clearTimeout(this.timer);
        this.timer = null;
      }
    },

    async step(auto) {
      if (this.busy || !this.started) return;
      this.busy = true;
      try {
        const payload = { auto };
        const data = await api("/api/decision/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        this.applyPayload(data);
        if (data.done) {
          this.done = true;
          this.stop();
          const result = data.state.success ? "成功到达目标" : (data.state.collision ? "发生碰撞" : "回合结束");
          this.setPlaceholder(`${result} · 累计奖励 ${data.total_return.toFixed(2)}`);
          $("#btn-step").disabled = true;
        }
      } catch (error) {
        this.stop();
        toast(error.message, true);
      } finally {
        this.busy = false;
      }
    },

    async manualStep(action, fromButton = false) {
      if (!this.started || this.done || this.policy !== "manual") {
        if (fromButton) toast("请选择「手动控制」策略并开始回合", true);
        return;
      }
      if (this.busy) return;
      this.busy = true;
      try {
        const data = await api("/api/decision/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ auto: false, action }),
        });
        this.applyPayload(data);
        if (data.done) {
          this.done = true;
          $("#btn-step").disabled = true;
          const result = data.state.success ? "成功到达目标" : (data.state.collision ? "发生碰撞" : "回合结束");
          this.setPlaceholder(`${result} · 累计奖励 ${data.total_return.toFixed(2)}`);
        }
      } catch (error) {
        toast(error.message, true);
      } finally {
        this.busy = false;
      }
    },

    onKey(event) {
      if (!$("#view-decision").classList.contains("active")) return;
      if (this.policy !== "manual" || !this.started) return;
      const keyMapNav = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };
      const keyMapCart = { ArrowLeft: 0, ArrowRight: 1 };
      const map = this.env === "robotnav" ? keyMapNav : keyMapCart;
      if (event.key in map) {
        event.preventDefault();
        this.manualStep(map[event.key]);
      }
    },

    applyPayload(data) {
      this.state = data.state;
      this.envType = data.state.type;
      this.updateStats(data);
      this.updateQValues(data);
      if (this.envType === "cartpole") {
        this.canvas.hidden = true;
        this.cartpoleImg.hidden = false;
        if (data.frame) this.cartpoleImg.src = data.frame;
      } else {
        this.canvas.hidden = false;
        this.cartpoleImg.hidden = true;
        this.drawGrid(data.state);
      }
      $("#episode-tag").textContent = `EP ${String(data.episode).padStart(2, "0")}`;
    },

    updateStats(data) {
      const s = data.state;
      $("#stat-return").textContent = Number(data.total_return).toFixed(2);
      $("#stat-steps").textContent = s.steps;
      if (s.type === "robotnav") {
        $("#stat-distance").textContent = s.distance;
      } else {
        $("#stat-distance").textContent = `${s.steps}/${s.max_steps}`;
      }
      $("#stat-reward").textContent = data.reward === null || data.reward === undefined
        ? "—" : Number(data.reward).toFixed(2);
    },

    updateQValues(data) {
      const container = $("#q-bars");
      if (!data.q_values || !data.q_values.length) {
        container.innerHTML = '<div class="empty small">当前策略不提供 Q 值</div>';
        return;
      }
      const labels = ACTION_LABELS[this.envType] || [];
      const values = data.q_values;
      const best = values.indexOf(Math.max(...values));
      const maxAbs = Math.max(1e-6, ...values.map((v) => Math.abs(v)));
      container.innerHTML = values.map((value, index) => {
        const width = Math.max(3, (Math.abs(value) / maxAbs) * 100);
        return `<div class="q-row ${index === best ? "best" : ""}">
          <span>${labels[index] || index}</span>
          <div class="q-track"><div class="q-fill" style="width:${width}%"></div></div>
          <span class="q-value">${value.toFixed(2)}</span>
        </div>`;
      }).join("");
    },

    drawEmptyGrid() {
      const ctx = this.ctx;
      const { width, height } = this.canvas;
      ctx.fillStyle = "#1c1e24";
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "#8b9bb0";
      ctx.font = "13px 'Microsoft YaHei', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("等待开始回合", width / 2, height / 2);
    },

    drawGrid(s) {
      const ctx = this.ctx;
      const width = this.canvas.width;
      const height = this.canvas.height;
      const n = s.grid_size;
      const pad = 24;
      const top = 18;
      const cell = Math.floor((Math.min(width, height - 100) - pad * 2) / n);
      const gridSize = cell * n;
      const ox = (width - gridSize) / 2;
      const oy = top;
      const toPx = (r, c) => [ox + c * cell, oy + r * cell];

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#1c1e24";
      ctx.fillRect(0, 0, width, height);

      // 网格
      ctx.strokeStyle = "#30343c";
      ctx.lineWidth = 1;
      for (let i = 0; i <= n; i += 1) {
        ctx.beginPath();
        ctx.moveTo(ox + i * cell, oy);
        ctx.lineTo(ox + i * cell, oy + gridSize);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(ox, oy + i * cell);
        ctx.lineTo(ox + gridSize, oy + i * cell);
        ctx.stroke();
      }

      // 静态障碍
      s.static.forEach(([r, c]) => {
        const [x, y] = toPx(r, c);
        ctx.fillStyle = "#464c58";
        roundRect(ctx, x + 3, y + 3, cell - 6, cell - 6, 5);
        ctx.fill();
      });

      // 轨迹
      s.trail.slice(0, -1).forEach(([r, c]) => {
        const [x, y] = toPx(r, c);
        ctx.fillStyle = "rgba(110, 140, 200, 0.35)";
        ctx.beginPath();
        ctx.arc(x + cell / 2, y + cell / 2, 3, 0, Math.PI * 2);
        ctx.fill();
      });

      // 目标
      const [gx, gy] = toPx(s.goal[0], s.goal[1]);
      ctx.fillStyle = "#5ac878";
      roundRect(ctx, gx + 6, gy + 6, cell - 12, cell - 12, 6);
      ctx.fill();
      ctx.fillStyle = "#12301c";
      ctx.font = `bold ${Math.floor(cell * 0.42)}px 'Segoe UI', sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("G", gx + cell / 2, gy + cell / 2 + 1);

      // 8 向射线（来自 11 维观测的前 8 维）
      const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]];
      const obs = s.observation || [];
      const [rx, ry] = toPx(s.robot[0], s.robot[1]);
      const cx = rx + cell / 2;
      const cy = ry + cell / 2;
      ctx.strokeStyle = "rgba(80, 200, 140, 0.13)";
      ctx.lineWidth = 1;
      for (let i = 0; i < Math.min(8, obs.length); i += 1) {
        const [dr, dc] = dirs[i];
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + dc * obs[i] * 6 * cell, cy + dr * obs[i] * 6 * cell);
        ctx.stroke();
      }

      // 动态障碍
      s.dynamic.forEach(([r, c]) => {
        const [x, y] = toPx(r, c);
        ctx.fillStyle = "#e05a3c";
        ctx.beginPath();
        ctx.arc(x + cell / 2, y + cell / 2, cell * 0.29, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.22)";
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // 机器人
      const [px, py] = toPx(s.robot[0], s.robot[1]);
      ctx.fillStyle = "#e8a04a";
      ctx.beginPath();
      ctx.arc(px + cell / 2, py + cell / 2, cell * 0.32, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#20242e";
      ctx.font = `bold ${Math.floor(cell * 0.36)}px 'Segoe UI', sans-serif`;
      const arrows = ["▲", "▼", "◀", "▶"];
      ctx.fillText(arrows[s.last_action] || "●", px + cell / 2, py + cell / 2 + 1);

      // 底部信息
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = "#c9d3e0";
      ctx.font = "12px 'Microsoft YaHei', sans-serif";
      const status = s.done ? (s.success ? "SUCCESS" : (s.collision ? "COLLISION" : "DONE")) : "RUNNING";
      ctx.fillText(`${status}   step ${s.steps}/${s.max_steps}`, ox, oy + gridSize + 28);
      ctx.fillStyle = "#8b9bb0";
      ctx.fillText(`距目标 ${s.distance} 格   最近奖励 ${Number(s.last_reward).toFixed(2)}`, ox, oy + gridSize + 48);

      // 图例
      ctx.fillStyle = "#8b9bb0";
      ctx.font = "10px 'Microsoft YaHei', sans-serif";
      const legendY = oy + gridSize + 72;
      ctx.fillStyle = "#e8a04a";
      ctx.beginPath(); ctx.arc(ox + 5, legendY - 4, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#8b9bb0"; ctx.fillText("机器人", ox + 14, legendY);
      ctx.fillStyle = "#e05a3c";
      ctx.beginPath(); ctx.arc(ox + 74, legendY - 4, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#8b9bb0"; ctx.fillText("动态障碍", ox + 83, legendY);
      ctx.fillStyle = "#5ac878";
      ctx.fillRect(ox + 154, legendY - 8, 8, 8);
      ctx.fillStyle = "#8b9bb0"; ctx.fillText("目标", ox + 167, legendY);
      ctx.fillStyle = "#464c58";
      ctx.fillRect(ox + 210, legendY - 8, 8, 8);
      ctx.fillStyle = "#8b9bb0"; ctx.fillText("静态障碍", ox + 223, legendY);
    },
  };

  function roundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  // ------------------------------------------------------------------
  // 初始化
  // ------------------------------------------------------------------
  function init() {
    $$(".nav-item").forEach((item) => {
      item.addEventListener("click", () => switchView(item.dataset.view));
    });
    $$("[data-goto]").forEach((item) => {
      item.addEventListener("click", () => {
        const target = item.dataset.goto;
        if (target === "detect") switchView("detect");
        else if (target === "face") switchView("face");
        else if (target === "depth") switchView("depth");
        else if (target === "decision") switchView("decision");
      });
    });

    state.analyzers.detect = new Analyzer("detect", "detect");
    state.analyzers.face = new Analyzer("face", "face");
    state.analyzers.depth = new Analyzer("depth", "depth,detect");

    decision.init();
    const initialView = location.hash.replace("#", "");
    if (PAGE_META[initialView]) switchView(initialView, false);
    window.__roboVision = { state, decision, switchView };
    loadHealth();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
