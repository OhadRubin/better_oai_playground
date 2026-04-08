import json
import html
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from openai import AsyncOpenAI

RUNS_DIR = Path(__file__).parent / "runs"

app = FastAPI()
client = AsyncOpenAI()

STREAMING_DELTA_TYPES = {
    "response.reasoning_summary_text.delta",
    "response.output_text.delta",
}
QUIET_TYPES = {
    "response.reasoning_summary_text.added",
    "response.reasoning_summary_text.done",
    "response.output_text.added",
    "response.output_text.done",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
}


# ── CSS ───────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }
a { color: #7aa2f7; text-decoration: none; }
a:hover { text-decoration: underline; }

.container { max-width: 1400px; margin: 0 auto; padding: 24px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid #2a2a3e; padding-bottom: 16px; }
.header h1 { font-size: 1.4rem; color: #c0caf5; }
.btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 500; }
.btn-primary { background: #7aa2f7; color: #0f0f1a; }
.btn-primary:hover { background: #89b4fa; }
.btn-danger { background: #f7768e; color: #0f0f1a; }
.btn-resume { background: #e0af68; color: #0f0f1a; }

/* Runs table */
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 10px 12px; color: #7a7a9e; font-weight: 500; font-size: 0.85rem; border-bottom: 1px solid #2a2a3e; }
td { padding: 10px 12px; border-bottom: 1px solid #1a1a2e; font-size: 0.9rem; }
tr:hover { background: #1a1a2e; }
.run-link { font-weight: 600; }

/* Status badges */
.badge { padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
.badge-streaming { background: #29426b; color: #7aa2f7; }
.badge-completed { background: #1a3a2a; color: #9ece6a; }
.badge-failed { background: #3a1a1a; color: #f7768e; }
.badge-queued { background: #3a3220; color: #e0af68; }
.badge-pending { background: #2a2a3e; color: #7a7a9e; }

/* New run form */
.form-group { margin-bottom: 16px; }
.form-group label { display: block; margin-bottom: 6px; color: #7a7a9e; font-size: 0.85rem; }
textarea { width: 100%; min-height: 500px; background: #1a1a2e; color: #c0caf5; border: 1px solid #2a2a3e; border-radius: 8px; padding: 16px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85rem; line-height: 1.6; resize: vertical; }
textarea:focus { outline: none; border-color: #7aa2f7; }

/* Run detail */
.detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.detail-panel { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 8px; overflow: hidden; }
.panel-header { padding: 10px 16px; background: #1e1e30; border-bottom: 1px solid #2a2a3e; font-size: 0.85rem; color: #7a7a9e; display: flex; justify-content: space-between; align-items: center; }
.panel-body { padding: 16px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.82rem; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; max-height: 80vh; overflow-y: auto; }

/* Streaming section headers */
.section-header { color: #e0af68; font-weight: 600; margin: 12px 0 4px 0; }

/* Meta bar */
.meta-bar { display: flex; gap: 16px; align-items: center; margin-bottom: 16px; padding: 12px 16px; background: #1a1a2e; border-radius: 8px; border: 1px solid #2a2a3e; font-size: 0.85rem; }
.meta-bar span { color: #7a7a9e; }
"""

# ── JS ────────────────────────────────────────────
JS_RUNS_LIST = """
document.addEventListener('DOMContentLoaded', async () => {
    function appendTextCell(row, text) {
        const td = document.createElement('td');
        td.textContent = text;
        row.appendChild(td);
        return td;
    }

    const resp = await fetch('/api/runs');
    const runs = await resp.json();
    const tbody = document.getElementById('runs-body');
    if (runs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#7a7a9e;padding:40px">No runs yet</td></tr>';
        return;
    }
    for (const run of runs) {
        const tr = document.createElement('tr');
        const promptPreview = run.prompt_preview ? run.prompt_preview.substring(0, 80) : '—';
        const runLinkCell = document.createElement('td');
        const runLink = document.createElement('a');
        runLink.href = '/runs/' + run.id;
        runLink.className = 'run-link';
        runLink.textContent = '#' + String(run.id).padStart(3, '0');
        runLinkCell.appendChild(runLink);
        tr.appendChild(runLinkCell);

        const statusCell = document.createElement('td');
        const badge = document.createElement('span');
        badge.className = 'badge badge-' + run.status;
        badge.textContent = run.status;
        statusCell.appendChild(badge);
        tr.appendChild(statusCell);

        appendTextCell(tr, run.created_at || '—');
        appendTextCell(tr, promptPreview);
        appendTextCell(
            tr,
            run.response_size === null || run.response_size === undefined
                ? '—'
                : run.response_size.toLocaleString() + ' chars'
        );
        tbody.appendChild(tr);
    }
});
"""

JS_NEW_RUN = """
async function submitRun() {
    const prompt = document.getElementById('prompt-input').value;
    if (!prompt.trim()) return;
    const btn = document.getElementById('submit-btn');
    btn.disabled = true;
    btn.textContent = 'Creating...';
    const resp = await fetch('/api/runs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt})
    });
    const data = await resp.json();
    window.location.href = '/runs/' + data.id;
}
"""

JS_RUN_DETAIL = """
let ws = null;
let reasoningArea, outputArea;

function appendToArea(area, text) {
    area.textContent += text;
    area.scrollTop = area.scrollHeight;
}

function replaceArea(area, text) {
    area.textContent = text;
    area.scrollTop = area.scrollHeight;
}

function currentStatus() {
    return document.getElementById('run-status').value;
}

function setRunStatus(status) {
    document.getElementById('run-status').value = status;
    const statusEl = document.getElementById('status-badge');
    statusEl.className = 'badge badge-' + status;
    statusEl.textContent = status;
    const actionBtn = document.getElementById('action-btn');
    if (!actionBtn) {
        return;
    }
    if (status === 'failed') {
        actionBtn.style.display = '';
        actionBtn.disabled = false;
        actionBtn.textContent = 'Retry';
        return;
    }
    if (status === 'queued' || status === 'streaming') {
        actionBtn.style.display = '';
        actionBtn.disabled = false;
        actionBtn.textContent = 'Reconnect';
        return;
    }
    actionBtn.style.display = 'none';
}

function closeSocketIfNeeded() {
    if (!ws) {
        return;
    }
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
    }
    ws = null;
}

function startStreaming(runId) {
    closeSocketIfNeeded();
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/runs/${runId}`);
    const actionBtn = document.getElementById('action-btn');
    if (actionBtn) {
        actionBtn.style.display = '';
        actionBtn.disabled = true;
        actionBtn.textContent = 'Connecting...';
    }

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
            case 'response.created':
                setRunStatus('streaming');
                break;
            case 'response.output_text.delta':
                appendToArea(outputArea, msg.delta);
                break;
            case 'response.output_text.full':
                replaceArea(outputArea, msg.text);
                break;
            case 'response.reasoning_summary_text.delta':
                appendToArea(reasoningArea, msg.delta);
                break;
            case 'response.completed':
                setRunStatus('completed');
                break;
            case 'error':
                setRunStatus('failed');
                appendToArea(outputArea, '\\n\\nERROR: ' + msg.message);
                break;
        }
    };
    ws.onclose = () => {
        const actionBtn = document.getElementById('action-btn');
        if (!actionBtn) {
            return;
        }
        const status = currentStatus();
        if (status === 'failed') {
            actionBtn.style.display = '';
            actionBtn.disabled = false;
            actionBtn.textContent = 'Retry';
            return;
        }
        if (status === 'queued' || status === 'streaming') {
            actionBtn.style.display = '';
            actionBtn.disabled = false;
            actionBtn.textContent = 'Reconnect';
        }
    };
}

async function resumeRun(runId) {
    const btn = document.getElementById('action-btn');
    if (!btn) {
        return;
    }
    btn.disabled = true;
    if (currentStatus() === 'failed') {
        btn.textContent = 'Retrying...';
        reasoningArea.textContent = '';
        outputArea.textContent = '';
        const resp = await fetch('/api/runs/' + runId + '/retry', {method: 'POST'});
        const data = await resp.json();
        if (!resp.ok) {
            setRunStatus('failed');
            appendToArea(outputArea, '\\n\\nERROR: ' + (data.error || 'Retry failed'));
            return;
        }
        setRunStatus(data.status);
    } else {
        btn.textContent = 'Reconnecting...';
    }
    startStreaming(runId);
}

document.addEventListener('DOMContentLoaded', () => {
    reasoningArea = document.getElementById('reasoning-area');
    outputArea = document.getElementById('output-area');
    const runId = document.getElementById('run-id').value;
    const status = currentStatus();
    setRunStatus(status);
    if (status === 'pending' || status === 'streaming' || status === 'queued') {
        startStreaming(runId);
    }
});
"""


# ── RunDir ────────────────────────────────────────
@dataclass
class RunDir:
    num: int

    @property
    def path(self) -> Path:
        return RUNS_DIR / f"{self.num:03d}"

    @property
    def prompt(self) -> Path:
        return self.path / "prompt.md"

    @property
    def response(self) -> Path:
        return self.path / "response.md"

    @property
    def meta_path(self) -> Path:
        return self.path / "meta.json"

    def load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        raise FileNotFoundError(f"No meta.json in run {self.num:03d}")

    def save_meta(self, data: dict):
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(data, indent=2))


def next_run_number() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(existing, default=0) + 1


def list_runs() -> list[dict]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    dirs = sorted(
        [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
        reverse=True,
    )
    for d in dirs:
        run = RunDir(int(d.name))
        meta = json.loads(run.meta_path.read_text()) if run.meta_path.exists() else {}
        prompt_preview = None
        if run.prompt.exists():
            text = run.prompt.read_text()
            first_line = text.strip().split("\n")[0] if text.strip() else ""
            prompt_preview = first_line
        response_size = None
        if run.response.exists():
            response_size = len(run.response.read_text())
        runs.append({
            "id": run.num,
            "status": meta.get("status", "pending"),
            "created_at": meta.get("created_at"),
            "prompt_preview": prompt_preview,
            "response_size": response_size,
        })
    return runs


def extract_text(resp) -> str:
    parts = []
    for item in resp.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    parts.append(content.text)
    return "\n\n".join(parts)


def format_traceback(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def fail_run(run: RunDir, meta: dict, message: str, exc: Exception | None = None):
    meta["status"] = "failed"
    meta["error"] = message
    if exc is None:
        meta.pop("traceback", None)
    else:
        meta["traceback"] = format_traceback(exc)
    run.save_meta(meta)


async def send_json_if_connected(ws: WebSocket, payload: dict) -> bool:
    try:
        await ws.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError:
        return False


async def send_completed(run: RunDir, meta: dict, ws: WebSocket, text: str, ws_connected: bool) -> bool:
    run.response.write_text(text)
    meta["status"] = "completed"
    meta.pop("error", None)
    meta.pop("traceback", None)
    run.save_meta(meta)
    if not ws_connected:
        return False
    ws_connected = await send_json_if_connected(ws, {"type": "response.output_text.full", "text": text})
    if not ws_connected:
        return False
    return await send_json_if_connected(ws, {"type": "response.completed", "status": "completed"})


def response_failure_message(resp) -> str:
    if getattr(resp, "error", None) is not None:
        return f"Response status: {resp.status} ({resp.error})"
    return f"Response status: {resp.status}"


# ── Core logic ────────────────────────────────────
async def stream_run(run: RunDir, ws: WebSocket):
    meta = run.load_meta()
    meta["status"] = "streaming"
    meta["response_id"] = None
    meta["cursor"] = None
    meta.pop("error", None)
    meta.pop("traceback", None)
    run.save_meta(meta)

    prompt = run.prompt.read_text()
    response_id = None
    ws_connected = True

    try:
        stream = await client.responses.create(
            model="gpt-5.4",
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            text={"format": {"type": "text"}, "verbosity": "high"},
            reasoning={"effort": "xhigh", "summary": "auto"},
            tools=[],
            store=True,
            include=["reasoning.encrypted_content", "web_search_call.action.sources"],
            background=True,
            stream=True,
        )

        async for event in stream:
            etype = event.type

            if response_id is None and hasattr(event, "response") and hasattr(event.response, "id"):
                response_id = event.response.id
                meta["response_id"] = response_id
                run.save_meta(meta)
                if ws_connected:
                    ws_connected = await send_json_if_connected(ws, {"type": "response.created", "model": "gpt-5.4"})

            if hasattr(event, "sequence_number") and event.sequence_number is not None:
                meta["cursor"] = event.sequence_number
                run.save_meta(meta)

            if etype in STREAMING_DELTA_TYPES:
                if ws_connected:
                    ws_connected = await send_json_if_connected(ws, {"type": etype, "delta": event.delta})
            elif etype == "response.completed":
                continue
            elif etype not in QUIET_TYPES and etype not in {"response.created", "response.in_progress"}:
                if ws_connected:
                    ws_connected = await send_json_if_connected(ws, {"type": etype})

        if not response_id:
            raise RuntimeError(f"Streaming finished without response_id for run {run.num:03d}")

        resp = await client.responses.retrieve(response_id)
        if resp.status == "completed":
            text = extract_text(resp)
            await send_completed(run, meta, ws, text, ws_connected)
            return
        if resp.status in {"queued", "in_progress"}:
            meta["status"] = "queued"
            run.save_meta(meta)
            await poll_and_stream(run, ws, response_id, ws_connected=ws_connected)
            return

        raise RuntimeError(response_failure_message(resp))
    except Exception as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        fail_run(run, meta, str(exc), exc)
        if ws_connected:
            await send_json_if_connected(ws, {"type": "error", "message": str(exc)})
        raise


async def poll_and_stream(run: RunDir, ws: WebSocket, response_id: str, ws_connected: bool = True):
    import asyncio

    meta = run.load_meta()
    while True:
        resp = await client.responses.retrieve(response_id)
        if resp.status not in {"queued", "in_progress"}:
            break
        await asyncio.sleep(2)

    if resp.status == "completed":
        text = extract_text(resp)
        await send_completed(run, meta, ws, text, ws_connected)
        return

    message = response_failure_message(resp)
    fail_run(run, meta, message)
    if ws_connected:
        await send_json_if_connected(ws, {"type": "error", "message": message})
    raise RuntimeError(message)


# ── API Endpoints ─────────────────────────────────
@app.get("/api/runs")
async def api_list_runs():
    return JSONResponse(list_runs())


@app.post("/api/runs")
async def api_create_run(body: dict):
    prompt_text = body["prompt"]
    run = RunDir(next_run_number())
    run.path.mkdir(parents=True, exist_ok=True)
    run.prompt.write_text(prompt_text)
    run.save_meta({
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "response_id": None,
        "cursor": None,
    })
    return JSONResponse({"id": run.num})


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: int):
    run = RunDir(run_id)
    if not run.path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = run.load_meta() if run.meta_path.exists() else {}
    prompt_text = run.prompt.read_text() if run.prompt.exists() else ""
    response_text = run.response.read_text() if run.response.exists() else ""
    return JSONResponse({
        "id": run.num,
        "status": meta.get("status", "pending"),
        "created_at": meta.get("created_at"),
        "prompt": prompt_text,
        "response": response_text,
        "response_id": meta.get("response_id"),
    })


@app.post("/api/runs/{run_id}/retry")
async def api_retry_run(run_id: int):
    run = RunDir(run_id)
    if not run.path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    meta = run.load_meta()
    if meta.get("status") != "failed":
        return JSONResponse({"error": f"cannot retry run with status {meta.get('status')}"}, status_code=409)

    if run.response.exists():
        run.response.unlink()

    meta["status"] = "pending"
    meta["response_id"] = None
    meta["cursor"] = None
    meta.pop("error", None)
    meta.pop("traceback", None)
    run.save_meta(meta)
    return JSONResponse({"id": run.num, "status": "pending"})


@app.websocket("/ws/runs/{run_id}")
async def ws_run(run_id: int, websocket: WebSocket):
    await websocket.accept()
    run = RunDir(run_id)
    try:
        meta = run.load_meta()
        if meta.get("status") == "pending":
            await stream_run(run, websocket)
        elif meta.get("status") in {"streaming", "queued"} and meta.get("response_id"):
            await poll_and_stream(run, websocket, meta["response_id"])
        elif meta.get("status") == "completed":
            await websocket.send_json({"type": "response.completed", "status": "completed"})
        else:
            await websocket.send_json({"type": "error", "message": f"Run status: {meta.get('status')}"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        should_send_error = True
        if run.path.exists() and run.meta_path.exists():
            meta = run.load_meta()
            if meta.get("status") == "failed" and meta.get("error") == str(exc):
                should_send_error = False
            else:
                fail_run(run, meta, str(exc), exc)
        if should_send_error:
            await send_json_if_connected(websocket, {"type": "error", "message": str(exc)})
        raise


# ── HTML Routes ───────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def page_runs_list():
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>resp — runs</title><style>{CSS}</style></head>
<body>
<div class="container">
    <div class="header">
        <h1>resp</h1>
        <a href="/new" class="btn btn-primary">New Run</a>
    </div>
    <table>
        <thead><tr>
            <th>Run</th><th>Status</th><th>Created</th><th>Prompt</th><th>Response</th>
        </tr></thead>
        <tbody id="runs-body"></tbody>
    </table>
</div>
<script>{JS_RUNS_LIST}</script>
</body>
</html>"""


@app.get("/new", response_class=HTMLResponse)
async def page_new_run():
    latest_prompt = ""
    runs = list_runs()
    if runs:
        latest = RunDir(runs[0]["id"])
        if latest.prompt.exists():
            latest_prompt = latest.prompt.read_text()
    escaped = html.escape(latest_prompt)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>resp — new run</title><style>{CSS}</style></head>
<body>
<div class="container">
    <div class="header">
        <h1><a href="/">resp</a> / new run</h1>
    </div>
    <div class="form-group">
        <label>Prompt</label>
        <textarea id="prompt-input">{escaped}</textarea>
    </div>
    <button class="btn btn-primary" id="submit-btn" onclick="submitRun()">Submit</button>
</div>
<script>{JS_NEW_RUN}</script>
</body>
</html>"""


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def page_run_detail(run_id: int):
    run = RunDir(run_id)
    if not run.path.exists():
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    meta = run.load_meta() if run.meta_path.exists() else {}
    status = meta.get("status", "pending")
    badge_class = f"badge-{status}"

    prompt_text = html.escape(run.prompt.read_text()) if run.prompt.exists() else ""
    response_text = html.escape(run.response.read_text()) if run.response.exists() else ""

    action_btn = ""
    if status in {"queued", "streaming"}:
        action_btn = f'<button class="btn btn-resume" id="action-btn" onclick="resumeRun({run_id})">Reconnect</button>'
    elif status == "failed":
        action_btn = f'<button class="btn btn-resume" id="action-btn" onclick="resumeRun({run_id})">Retry</button>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>resp — run #{run_id:03d}</title><style>{CSS}</style></head>
<body>
<input type="hidden" id="run-id" value="{run_id}">
<input type="hidden" id="run-status" value="{status}">
<div class="container">
    <div class="header">
        <h1><a href="/">resp</a> / run #{run_id:03d}</h1>
        <div style="display:flex;gap:12px;align-items:center">
            <span class="badge {badge_class}" id="status-badge">{status}</span>
            {action_btn}
        </div>
    </div>
    <div class="meta-bar">
        <span>Created: {meta.get("created_at", "—")}</span>
        <span>Response ID: {meta.get("response_id", "—")}</span>
    </div>
    <div class="detail-grid">
        <div class="detail-panel">
            <div class="panel-header">Prompt</div>
            <div class="panel-body">{prompt_text}</div>
        </div>
        <div class="detail-panel">
            <div class="panel-header">
                <span>Response</span>
                <span>Reasoning</span>
            </div>
            <div class="panel-body" id="reasoning-area" style="max-height:30vh;border-bottom:1px solid #2a2a3e;color:#7a7a9e;font-size:0.78rem">{""}</div>
            <div class="panel-body" id="output-area">{response_text}</div>
        </div>
    </div>
</div>
<script>{JS_RUN_DETAIL}</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8042)
