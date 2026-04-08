<template>
  <div>
    <div class="toolbar">
      <span>方法</span>
      <select v-model="methodTitle" style="width: 190px">
        <option v-for="m in methodTitles" :key="m" :value="m">{{ m }}</option>
      </select>
      <span>ID</span>
      <input v-model="sampleId" style="width: 320px" />
      <button @click="connect">连接</button>
      <button type="button" @click="fitViewToNodes">居中显示全图</button>
      <span class="badge">{{ status }}</span>
      <span class="muted" style="font-size: 12px; margin-left: 8px">
        在空白处拖拽平移；节点多时请点「居中」或向右拖看后续 hop
      </span>
    </div>

    <div class="layout">
      <div class="canvas" ref="canvasEl" :class="{ dragging: isDragging }">
        <div
          class="canvas-pan"
          :style="{ transform: `translate(${panX}px, ${panY}px)` }"
        >
        <svg
          :width="svgW"
          :height="svgH"
          style="display: block"
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#7382a8" />
            </marker>
          </defs>

          <!-- edges -->
          <g>
            <path
              v-for="e in edges"
              :key="e.id"
              :d="edgePath(e)"
              stroke="#7382a8"
              stroke-width="2"
              fill="none"
              marker-end="url(#arrow)"
              opacity="0.95"
            />
          </g>

          <!-- nodes -->
          <g v-for="n in nodes" :key="n.id" @click="selectNode(n)" style="cursor: pointer">
            <!-- outer ring (rotating) -->
            <circle
              :cx="n.x"
              :cy="n.y"
              :r="nodeR + 10"
              fill="none"
              stroke="rgba(255,255,255,0.22)"
              stroke-width="3"
              stroke-dasharray="10 8"
              :class="n.thinking ? 'thinkingRing' : ''"
            />

            <!-- base circle -->
            <circle
              :cx="n.x"
              :cy="n.y"
              :r="nodeR"
              :fill="roleColor(n.role)"
              :stroke="roleBorder(n.role)"
              stroke-width="3"
              class="nodeBase"
              :class="n.thinking ? 'thinkingGlow' : ''"
            />

            <!-- label -->
            <text
              :x="n.x"
              :y="n.y"
              text-anchor="middle"
              dominant-baseline="middle"
              fill="#fff"
              font-size="14"
              style="pointer-events: none"
            >
              {{ n.baseLabel }}
            </text>
          </g>
        </svg>
        </div>
      </div>

      <div class="panel">
        <div style="font-weight: 700; margin-bottom: 8px">节点详情</div>
        <div v-if="!selected" class="muted">点击左侧节点查看 AI 输出</div>
        <div v-else>
          <div>
            <b>{{ selected.baseLabel }}</b>
          </div>
          <div class="muted" style="margin-top: 4px">
            role={{ selected.role || "" }} | hop={{ selected.hop }} | phase={{ selected.phase }}
          </div>

          <div style="margin-top: 10px"><b>结论</b></div>
          <pre>{{ selected.conclusion || "(空)" }}</pre>

          <details>
            <summary>AI 思考内容（可折叠）</summary>
            <pre>{{ selected.thinkingText || "(无可用原始输出)" }}</pre>
          </details>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref, computed, watch, nextTick } from "vue";
import { fetchEvents, fetchRuns } from "./api";

const methodTitles = ["IO", "COT", "TOT", "GOT", "multiAgentGoT"];
const methodTitle = ref("multiAgentGoT");
const sampleId = ref("");
const status = ref("未连接");

const lastSeq = ref(0);
let timer = null;

const nodeR = 62;
const dx = 260;
const dy = 140;
const marginX = 120;
const marginY = 110;

const opColumn = new Map(); // op_id -> col
const opRows = new Map(); // op_id -> next row
const thinkingOps = new Set(); // op_id active

const nodesById = new Map();
const edgesById = new Map();

const nodes = ref([]);
const edges = ref([]);
const selected = ref(null);
const canvasEl = ref(null);
const isDragging = ref(false);
/** 画布平移（无滚动条，仅拖拽） */
const panX = ref(0);
const panY = ref(0);

/** 按节点实际坐标包络计算，避免纵向/横向堆叠超出固定高度后被裁切（看不到后续 hop） */
const svgW = computed(() => {
  const pad = nodeR + 48;
  const list = nodes.value;
  if (!list.length) return 1400;
  const maxX = Math.max(...list.map((n) => n.x));
  return Math.max(1200, Math.ceil(maxX + pad));
});
const svgH = computed(() => {
  const pad = nodeR + 48;
  const list = nodes.value;
  if (!list.length) return 800;
  const maxY = Math.max(...list.map((n) => n.y));
  return Math.max(700, Math.ceil(maxY + pad));
});

function titleToMethodName(t) {
  const s = String(t || "").trim().toLowerCase();
  if (s === "io") return "io";
  if (s === "cot") return "cot";
  if (s === "tot") return "tot";
  if (s === "got") return "got";
  if (s === "multiagentgot") return "multiAgentGoT";
  return "multiAgentGoT";
}

function runId() {
  const sid = sampleId.value.trim();
  if (!sid) return "";
  return `${titleToMethodName(methodTitle.value)}:${sid}`;
}

function roleColor(role) {
  const r = String(role || "");
  if (r === "planner") return "#f59e0b";
  if (r === "retriever") return "#22c55e";
  if (r === "reasoner") return "#3b82f6";
  if (r === "critic" || r === "critic_done") return "#ef4444";
  if (r === "aggregate" || r === "aggregator" || r === "final_answer") return "#ec4899";
  if (r === "advance") return "#14b8a6";
  if (r === "selector") return "#64748b";
  return "#8b5cf6";
}
function roleBorder(role) {
  const r = String(role || "");
  if (r === "planner") return "#fde68a";
  if (r === "retriever") return "#bbf7d0";
  if (r === "reasoner") return "#bfdbfe";
  if (r === "critic" || r === "critic_done") return "#fecaca";
  if (r === "aggregate" || r === "aggregator" || r === "final_answer") return "#fbcfe8";
  if (r === "advance") return "#ccfbf1";
  if (r === "selector") return "#cbd5e1";
  return "#d6c3ff";
}

function colForNode(n) {
  const method = titleToMethodName(methodTitle.value);
  if (method === "multiAgentGoT") {
    const op = n.op_type || "";
    const phase = n.phase;
    const hop = n.hop >= 0 ? n.hop : 0;
    
    // Planner
    if (op === "generate" && phase === 1) return 0;
    
    // All hops share the same column offsets to keep them vertically aligned
    const hopOffset = 1;
    
    if (op === "generate" && phase === 2) return hopOffset;
    if (op === "score" && phase === 2) return hopOffset + 1;
    if (op === "keep_best_n" && phase === 2) return hopOffset + 2;
    
    if (op === "generate" && phase === 3) return hopOffset + 3;
    if (op === "score" && phase === 3) return hopOffset + 4;
    if (op === "keep_best_n" && phase === 3) return hopOffset + 5;
    
    if (op === "critic_verify_and_backtrack") return hopOffset + 6;
    if (op === "advance_subquestion") return hopOffset + 7;
    
    // Final aggregation phase
    if (op === "selector") return 9;
    if (op === "aggregate") return 10;
    if (op === "score" && phase === 4) return 11;
    if (op === "keep_best_n" && phase === 4) return 12;
    if (op === "ground_truth_evaluator") return 13;
  }
  const k = String(n.op_id);
  if (!opColumn.has(k)) opColumn.set(k, opColumn.size);
  return opColumn.get(k);
}

function nextRow(colIndex, n) {
  const method = titleToMethodName(methodTitle.value);
  if (method === "multiAgentGoT") {
    // Center planner and final nodes vertically (e.g., around row 5)
    if ((n.op_type === "generate" && n.phase === 1) || 
        n.op_type === "selector" || 
        n.op_type === "aggregate" || 
        (n.op_type === "score" && n.phase === 4) || 
        (n.op_type === "keep_best_n" && n.phase === 4) || 
        n.op_type === "ground_truth_evaluator") {
        
        const k = String(colIndex) + "_centered";
        const count = opRows.get(k) || 0;
        opRows.set(k, count + 1);
        return 5 + count;
    }
    
    let hop = n.hop >= 0 ? n.hop : 0;
    
    // Advance node has already incremented its hop in the state, 
    // so we subtract 1 to align it with the row of the hop it originated from.
    if (n.op_type === "advance_subquestion") {
        hop = Math.max(0, hop - 1);
    }
    
    // Base row offset based on hop to separate hops vertically
    const baseRow = hop * 3; 
    
    const k = String(colIndex) + "_" + String(hop);
    const count = opRows.get(k) || 0;
    opRows.set(k, count + 1);
    
    return baseRow + count;
  }
  
  const k = String(colIndex);
  const count = opRows.get(k) || 0;
  opRows.set(k, count + 1);
  return count;
}

function resetGraph() {
  lastSeq.value = 0;
  opColumn.clear();
  opRows.clear();
  thinkingOps.clear();
  nodesById.clear();
  edgesById.clear();
  nodes.value = [];
  edges.value = [];
  selected.value = null;
  panX.value = 0;
  panY.value = 0;
}

function selectNode(n) {
  selected.value = n;
}

/** 将图心移到画布中心，便于看到右侧后续 hop 或纵向堆叠的节点 */
function fitViewToNodes() {
  const list = nodes.value;
  if (!list.length) return;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const n of list) {
    minX = Math.min(minX, n.x - nodeR - 12);
    maxX = Math.max(maxX, n.x + nodeR + 12);
    minY = Math.min(minY, n.y - nodeR - 12);
    maxY = Math.max(maxY, n.y + nodeR + 12);
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const el = canvasEl.value;
  const cw = el ? el.clientWidth : 800;
  const ch = el ? el.clientHeight : 600;
  panX.value = Math.round(cw / 2 - cx);
  panY.value = Math.round(ch / 2 - cy);
}

function edgePath(e) {
  const s = nodesById.get(e.source);
  const t = nodesById.get(e.target);
  if (!s || !t) return "";
  const x1 = s.x;
  const y1 = s.y;
  const x2 = t.x;
  const y2 = t.y;
  const mx = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
}

function ensureEdge(sourceId, targetId, idx) {
  const id = `e_${sourceId}_${targetId}_${idx}`;
  if (edgesById.has(id)) return;
  const e = { id, source: sourceId, target: targetId };
  edgesById.set(id, e);
  edges.value = Array.from(edgesById.values());
}

function ensureNode(node, opId, opType, parentIds) {
  const id = node.id;
  if (nodesById.has(id)) return;
  const col = colForNode(node);
  const row = nextRow(col, node);
  const x = marginX + col * dx;
  const y = marginY + row * dy;

  const n = {
    id,
    baseLabel: node.label || id,
    role: node.role || "",
    hop: node.hop ?? -1,
    phase: node.phase ?? -1,
    op_id: opId,
    op_type: opType,
    x,
    y,
    // animation flag is controlled by op_start/op_end
    thinking: thinkingOps.has(String(opId)),
    conclusion: node.conclusion || "",
    thinkingText: String(node.thinking || ""),
  };

  nodesById.set(id, n);
  nodes.value = Array.from(nodesById.values());

  // Connect to planner if this is a hop's first retriever
  const method = titleToMethodName(methodTitle.value);
  if (method === "multiAgentGoT" && opType === "generate" && node.phase === 2) {
    const plannerNode = Array.from(nodesById.values()).find(
      (n) => n.op_type === "generate" && n.phase === 1
    );
    if (plannerNode) {
      ensureEdge(plannerNode.id, id, 0);
    }
  }

  // Draw regular edges
  (parentIds || []).forEach((pid, idx) => {
    // If this is a hop > 0 retriever, do NOT draw the edge from the previous hop's advance node
    // because we already drew the edge from the planner. This keeps the graph from looking like a snake.
    if (method === "multiAgentGoT" && opType === "generate" && node.phase === 2 && node.hop > 0) {
      const parentNode = nodesById.get(pid);
      if (parentNode && parentNode.op_type === "advance_subquestion") {
        return; // Skip this edge
      }
    }
    ensureEdge(pid, id, idx + 1);
  });
}

function setThinking(opId, active) {
  const k = String(opId);
  if (active) thinkingOps.add(k);
  else thinkingOps.delete(k);
  // update nodes currently linked to this op
  nodesById.forEach((n) => {
    if (String(n.op_id) === k) n.thinking = thinkingOps.has(k);
  });
  nodes.value = Array.from(nodesById.values());
}

function handleEvent(payload) {
  if (!payload || !payload.type) return;
  if (payload.type === "op_start") setThinking(payload.op_id, true);
  if (payload.type === "op_end") setThinking(payload.op_id, false);
  if (payload.type === "thought_created") {
    ensureNode(payload.node, payload.op_id, payload.op_type, payload.parent_ids || []);
  }
  if (payload.type === "run_end") {
    status.value = "运行结束";
    nextTick(() => fitViewToNodes());
  }
}

async function poll() {
  const rid = runId();
  if (!rid) return;
  try {
    const data = await fetchEvents(rid, lastSeq.value);
    const evs = data.events || [];
    evs.forEach((e) => {
      lastSeq.value = Math.max(lastSeq.value, Number(e.seq || 0));
      handleEvent(e.payload || {});
    });
    if (status.value !== "运行结束") status.value = "实时中";
  } catch (e) {
    status.value = "连接失败";
  }
}

function connect() {
  resetGraph();
  status.value = "连接中...";
  if (timer) clearInterval(timer);
  timer = setInterval(poll, 350);
  poll();
}

async function initDefault() {
  try {
    const data = await fetchRuns();
    const latest = data.latest || {};
    if (latest.method) {
      // method here is like "multiAgentGoT"
      methodTitle.value = latest.method === "multiAgentGoT" ? "multiAgentGoT" : latest.method.toUpperCase();
    }
    if (latest.sample_id) {
      sampleId.value = String(latest.sample_id);
    }
  } catch (_) {
    // ignore
  }
}

watch(methodTitle, async () => {
  // change method => fill latest id for that method
  try {
    const data = await fetchRuns();
    const by = data.by_method || {};
    const m = titleToMethodName(methodTitle.value);
    const arr = by[m] || [];
    sampleId.value = arr.length ? String(arr[arr.length - 1]) : "";
  } catch (_) {
    // ignore
  }
});

onMounted(() => {
  initDefault();
  setupCanvasDragPan();
});

function setupCanvasDragPan() {
  const el = canvasEl.value;
  if (!el) return;

  let startX = 0;
  let startY = 0;
  let startPanX = 0;
  let startPanY = 0;

  const onDown = (e) => {
    if (e.button !== 0) return;
    const tag = e.target && e.target.tagName ? String(e.target.tagName).toLowerCase() : "";
    if (tag === "circle" || tag === "text" || tag === "path") return;
    isDragging.value = true;
    startX = e.clientX;
    startY = e.clientY;
    startPanX = panX.value;
    startPanY = panY.value;
  };

  const onMove = (e) => {
    if (!isDragging.value) return;
    panX.value = startPanX + (e.clientX - startX);
    panY.value = startPanY + (e.clientY - startY);
  };

  const onUp = () => {
    isDragging.value = false;
  };

  el.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}
</script>

