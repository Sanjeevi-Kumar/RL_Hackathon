"""
Warehouse RL Environment — FastAPI Server
==========================================
Exposes the OpenEnv standard HTTP API:

  POST /reset        → Observation  (start a new episode)
  POST /step         → Observation  (send one Action)
  GET  /state        → State        (full internal state for debugging)
  GET  /web          → HTML UI      (built-in browser interface)
  GET  /health       → dict         (liveness check)

Run with:
  uvicorn src.envs.warehouse_env.server.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from src.envs.warehouse_env.models import Action, Observation, State
from src.envs.warehouse_env.server.environment import WarehouseEnvironment

app = FastAPI(
    title="🏭 Warehouse RL Environment",
    description=(
        "OpenEnv-compliant RL environment for multi-product warehouse navigation. "
        "Built for the Meta OpenEnv AI Hackathon."
    ),
    version="1.0.0",
)

# Single global environment instance (stateful between steps)
_env = WarehouseEnvironment()


# ---------------------------------------------------------------------------
# OPENENV STANDARD ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation, summary="Start a new episode")
def reset() -> Observation:
    """
    Resets the warehouse to its initial state and returns the first observation.
    Call this at the start of every episode.
    """
    return _env.reset()


@app.post("/step", response_model=Observation, summary="Execute one action")
def step(action: Action) -> Observation:
    """
    Execute a single agent action and return the resulting observation.

    **Action fields:**
    - `movement`: north / south / east / west / stay
    - `interact`: pickup / deposit / recharge / none
    - `target_sku`: SKU to pick up (required when interact='pickup')
    """
    try:
        return _env.step(action)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Environment error: {exc}")


@app.get("/state", response_model=State, summary="Get full internal state")
def get_state() -> State:
    """
    Returns the full internal state (for debugging / RL framework inspection).
    This is more detailed than the Observation returned to the agent.
    """
    return _env.state


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok", "episode_id": _env.state.episode_id, "step": _env.state.step_count}


# ---------------------------------------------------------------------------
# BUILT-IN WEB UI  (openenv /web standard)
# ---------------------------------------------------------------------------

@app.get("/web", response_class=HTMLResponse, summary="Interactive browser UI")
def web_ui():
    """
    Built-in web interface for manually testing and inspecting the environment.
    Displays the warehouse grid, agent telemetry, product manifest, and action log.
    """
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🏭 Warehouse RL — OpenEnv UI</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2d3148;
    --accent: #6c63ff; --green: #22c55e; --red: #ef4444;
    --amber: #f59e0b; --text: #e2e8f0; --muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', monospace; }
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 12px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; }
  header .badge { font-size: 11px; background: var(--accent); color: #fff;
                  border-radius: 4px; padding: 2px 8px; }
  .layout { display: grid; grid-template-columns: auto 1fr; gap: 16px; padding: 16px; }
  /* Grid canvas */
  #warehouse-canvas { border: 1px solid var(--border); border-radius: 8px; display: block; }
  /* Right panel */
  .panel { background: var(--surface); border: 1px solid var(--border);
           border-radius: 8px; padding: 14px; }
  .panel h3 { font-size: 13px; font-weight: 600; color: var(--muted);
              text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }
  .metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px; }
  .metric { background: var(--bg); border-radius: 6px; padding: 8px 10px; }
  .metric .label { font-size: 10px; color: var(--muted); margin-bottom: 2px; }
  .metric .value { font-size: 18px; font-weight: 700; }
  .battery-bar { height: 6px; border-radius: 3px; background: var(--border); margin: 4px 0; }
  .battery-fill { height: 100%; border-radius: 3px; transition: width .3s; }
  /* Controls */
  .controls { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin: 10px 0; }
  button { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
           color: var(--text); padding: 8px 12px; cursor: pointer; font-size: 13px;
           transition: background .15s; }
  button:hover { background: var(--border); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.primary:hover { opacity: .85; }
  button.danger { border-color: var(--red); color: var(--red); }
  /* Action form */
  .form-row { display: flex; gap: 8px; margin: 6px 0; align-items: center; }
  select, input { background: var(--bg); border: 1px solid var(--border); border-radius: 5px;
                  color: var(--text); padding: 6px 10px; font-size: 12px; flex: 1; }
  /* Log */
  #log { height: 140px; overflow-y: auto; background: var(--bg); border-radius: 6px;
         padding: 8px; font-size: 11px; font-family: monospace; border: 1px solid var(--border); }
  .log-ok   { color: var(--green); }
  .log-err  { color: var(--red); }
  .log-warn { color: var(--amber); }
  /* Product list */
  #product-list { font-size: 11px; max-height: 160px; overflow-y: auto; }
  .prod-row { display: flex; justify-content: space-between; align-items: center;
              padding: 3px 0; border-bottom: 1px solid var(--border); }
  .prod-row:last-child { border: none; }
  .tag { font-size: 10px; border-radius: 3px; padding: 1px 6px; font-weight: 600; }
  .tag-high   { background: #ef444422; color: var(--red); }
  .tag-medium { background: #f59e0b22; color: var(--amber); }
  .tag-low    { background: #22c55e22; color: var(--green); }
  .tag-done   { background: #33333366; color: var(--muted); text-decoration: line-through; }
  .hint { background: #6c63ff11; border: 1px solid #6c63ff44; border-radius: 6px;
          padding: 8px 10px; font-size: 12px; color: #a5b4fc; margin: 8px 0; }
  .right-col { display: flex; flex-direction: column; gap: 12px; }
</style>
</head>
<body>
<header>
  <span style="font-size:22px">🏭</span>
  <h1>Warehouse RL Environment</h1>
  <span class="badge">OpenEnv v1.0</span>
  <span class="badge" style="background:#22c55e" id="status-badge">READY</span>
</header>

<div class="layout">
  <!-- Warehouse grid -->
  <div>
    <canvas id="warehouse-canvas" width="480" height="480"></canvas>
    <div style="font-size:11px;color:var(--muted);margin-top:6px;text-align:center">
      ● Agent &nbsp; ★ Dock &nbsp; ⚡ Recharge &nbsp; ■ Shelf &nbsp; ⬛ Collision &nbsp;
      🔴 High &nbsp; 🟠 Med &nbsp; 🟢 Low
    </div>
  </div>

  <!-- Right panel -->
  <div class="right-col">
    <!-- Metrics -->
    <div class="panel">
      <h3>Agent Status</h3>
      <div class="metrics">
        <div class="metric"><div class="label">Steps</div><div class="value" id="m-steps">0</div></div>
        <div class="metric"><div class="label">Score</div><div class="value" id="m-score">0.0</div></div>
        <div class="metric"><div class="label">Deposited</div><div class="value" id="m-dep">0/15</div></div>
      </div>
      <div style="font-size:11px;color:var(--muted)">Battery</div>
      <div class="battery-bar"><div class="battery-fill" id="battery-fill" style="width:100%;background:var(--green)"></div></div>
      <div style="font-size:11px;color:var(--muted)" id="battery-text">100%</div>
      <div style="font-size:11px;margin-top:6px">
        Position: <span id="m-pos" style="color:var(--accent)">(11,0)</span> &nbsp;
        Inventory: <span id="m-inv" style="color:var(--amber)">empty</span>
      </div>
    </div>

    <!-- Controls -->
    <div class="panel">
      <h3>Manual Control</h3>
      <div class="controls">
        <div></div>
        <button onclick="sendAction('north','none')">⬆ North</button>
        <div></div>
        <button onclick="sendAction('west','none')">⬅ West</button>
        <button onclick="sendAction('stay','none')">⏹ Stay</button>
        <button onclick="sendAction('east','none')">➡ East</button>
        <div></div>
        <button onclick="sendAction('south','none')">⬇ South</button>
        <div></div>
      </div>
      <div class="form-row" style="margin-top:8px">
        <select id="sel-interact">
          <option value="none">No interact</option>
          <option value="pickup">📦 Pickup</option>
          <option value="deposit">🚚 Deposit</option>
          <option value="recharge">⚡ Recharge</option>
        </select>
        <select id="sel-move">
          <option value="stay">Stay</option>
          <option value="north">North</option>
          <option value="south">South</option>
          <option value="east">East</option>
          <option value="west">West</option>
        </select>
      </div>
      <div class="form-row">
        <input id="inp-sku" placeholder="SKU (for pickup, e.g. SKU-H1)" />
        <button class="primary" onclick="sendCustomAction()">▶ Send</button>
      </div>
      <div class="form-row">
        <button style="flex:1" onclick="doReset()">↺ Reset Episode</button>
      </div>
    </div>

    <!-- Hint -->
    <div class="hint" id="hint-box">Press Reset to start a new episode.</div>

    <!-- Product manifest -->
    <div class="panel" style="flex:1">
      <h3>Product Manifest</h3>
      <div id="product-list">Loading...</div>
    </div>

    <!-- Log -->
    <div class="panel">
      <h3>Action Log</h3>
      <div id="log"></div>
    </div>
  </div>
</div>

<script>
const ROWS = 12, COLS = 12, CELL = 40;
const canvas = document.getElementById('warehouse-canvas');
const ctx = canvas.getContext('2d');
let obs = null;

const COLORS = {
  bg: '#0f1117', surface: '#1a1d27', border: '#2d3148',
  shelf: '#334155', collision: '#1e1b2e', empty: '#16192b',
  agent: '#6c63ff', dock: '#22c55e', recharge: '#06b6d4',
  high: '#ef4444', medium: '#f59e0b', low: '#22c55e',
  collected: '#475569', text: '#e2e8f0',
};

function drawGrid() {
  if (!obs) return;
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw grid lines
  ctx.strokeStyle = '#1e2135';
  ctx.lineWidth = 0.5;
  for (let r = 0; r <= ROWS; r++) {
    ctx.beginPath(); ctx.moveTo(0, r*CELL); ctx.lineTo(COLS*CELL, r*CELL); ctx.stroke();
  }
  for (let c = 0; c <= COLS; c++) {
    ctx.beginPath(); ctx.moveTo(c*CELL, 0); ctx.lineTo(c*CELL, ROWS*CELL); ctx.stroke();
  }

  // Overlay visible cells
  if (obs.visible_cells) {
    obs.visible_cells.forEach(cell => {
      const [r, c] = cell.position;
      const x = c * CELL, y = r * CELL;
      let fill = null;
      if (cell.kind === 'shelf_obstacle') fill = COLORS.shelf;
      else if (cell.kind === 'collision_zone') fill = COLORS.collision;
      else if (cell.kind === 'loading_dock') fill = '#052e16';
      else if (cell.kind === 'recharge_station') fill = '#083344';

      if (fill) {
        ctx.fillStyle = fill;
        ctx.fillRect(x+1, y+1, CELL-2, CELL-2);
      }
    });
  }

  // Draw products from manifest
  obs.products.forEach(p => {
    if (p.collected || !p.position) return;
    const [r, c] = p.position;
    const x = c * CELL, y = r * CELL;
    ctx.fillStyle = COLORS[p.priority] + '33';
    ctx.fillRect(x+1, y+1, CELL-2, CELL-2);
    ctx.fillStyle = COLORS[p.priority];
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(p.sku.replace('SKU-',''), x+CELL/2, y+CELL/2);
  });

  // Loading dock
  const [dr, dc] = obs.loading_dock_position;
  ctx.fillStyle = COLORS.dock + '33';
  ctx.fillRect(dc*CELL+1, dr*CELL+1, CELL-2, CELL-2);
  ctx.fillStyle = COLORS.dock;
  ctx.font = '18px sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('★', dc*CELL+CELL/2, dr*CELL+CELL/2);

  // Recharge stations
  obs.recharge_stations.forEach(([rr,rc]) => {
    ctx.fillStyle = COLORS.recharge + '33';
    ctx.fillRect(rc*CELL+1, rr*CELL+1, CELL-2, CELL-2);
    ctx.fillStyle = COLORS.recharge;
    ctx.font = '16px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('⚡', rc*CELL+CELL/2, rr*CELL+CELL/2);
  });

  // Agent
  const [ar, ac] = obs.agent_position;
  ctx.fillStyle = COLORS.agent;
  const r = CELL * 0.35;
  ctx.beginPath();
  ctx.arc(ac*CELL+CELL/2, ar*CELL+CELL/2, r, 0, Math.PI*2);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('R', ac*CELL+CELL/2, ar*CELL+CELL/2);
}

function updateUI() {
  if (!obs) return;
  document.getElementById('m-steps').textContent = obs.steps_taken;
  document.getElementById('m-score').textContent = obs.score.toFixed(2);
  document.getElementById('m-dep').textContent = `${obs.products_deposited}/${obs.total_products}`;
  document.getElementById('m-pos').textContent = JSON.stringify(obs.agent_position);
  document.getElementById('m-inv').textContent = obs.inventory.length ? obs.inventory.join(', ') : 'empty';

  const batt = Math.round(obs.battery_level * 100);
  document.getElementById('battery-text').textContent = batt + '%';
  const fill = document.getElementById('battery-fill');
  fill.style.width = batt + '%';
  fill.style.background = batt > 50 ? '#22c55e' : batt > 20 ? '#f59e0b' : '#ef4444';

  document.getElementById('hint-box').textContent = obs.hint;
  document.getElementById('status-badge').textContent = obs.done ? (obs.is_success ? '✅ SUCCESS' : '❌ DONE') : 'RUNNING';
  document.getElementById('status-badge').style.background = obs.done ? (obs.is_success ? '#22c55e' : '#ef4444') : '#6c63ff';

  // Products
  const list = document.getElementById('product-list');
  list.innerHTML = obs.products.map(p => {
    const tagClass = p.deposited ? 'tag-done' : `tag-${p.priority}`;
    const status = p.deposited ? 'deposited' : p.collected ? 'carrying' : 'available';
    return `<div class="prod-row">
      <span class="tag ${tagClass}">${p.priority.toUpperCase()}</span>
      <span style="font-weight:600">${p.sku}</span>
      <span style="color:var(--muted)">${JSON.stringify(p.position)}</span>
      <span style="color:var(--muted)">${status}</span>
      <span style="color:var(--amber)">+${(p.value).toFixed(1)}</span>
    </div>`;
  }).join('');

  drawGrid();
}

function log(msg, kind='ok') {
  const div = document.getElementById('log');
  const entry = document.createElement('div');
  entry.className = `log-${kind}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  div.prepend(entry);
  while (div.children.length > 60) div.removeChild(div.lastChild);
}

async function doReset() {
  const res = await fetch('/reset', { method: 'POST' });
  obs = await res.json();
  updateUI();
  log('Episode reset.', 'ok');
}

async function sendAction(movement, interact, sku=null) {
  const body = { movement, interact, target_sku: sku, metadata: {} };
  const res = await fetch('/step', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
  });
  obs = await res.json();
  updateUI();
  const kind = obs.last_action_valid ? (obs.reward > 0 ? 'ok' : 'warn') : 'err';
  log(`${movement} | ${interact} | r=${obs.reward.toFixed(3)} | ${obs.last_action_message}`, kind);
}

async function sendCustomAction() {
  const movement = document.getElementById('sel-move').value;
  const interact = document.getElementById('sel-interact').value;
  const sku = document.getElementById('inp-sku').value.trim() || null;
  await sendAction(movement, interact, sku);
}

// Initial load
doReset();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
