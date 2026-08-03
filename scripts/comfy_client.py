"""Minimal ComfyUI API client: queue an API-format workflow and wait for it.

Also samples GPU memory while the job runs, because peak VRAM during real
generation is the number that actually matters for planning, and it cannot be
read after the fact.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188, logger=None):
        self.base = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())
        self.log = logger

    # -- plumbing -----------------------------------------------------------
    def _get(self, path: str, timeout: int = 30):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read())

    def _post(self, path: str, payload: dict, timeout: int = 60):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise ComfyError(f"POST {path} -> HTTP {e.code}\n{body}") from None

    # -- public -------------------------------------------------------------
    def is_up(self) -> bool:
        try:
            self._get("/system_stats", timeout=5)
            return True
        except Exception:
            return False

    def object_info(self) -> dict:
        return self._get("/object_info")

    def queue(self, api_workflow: dict) -> str:
        res = self._post("/prompt", {"prompt": api_workflow,
                                     "client_id": self.client_id})
        if "prompt_id" not in res:
            raise ComfyError(f"No prompt_id in response: {res}")
        return res["prompt_id"]

    def validate(self, api_workflow: dict) -> tuple[bool, str]:
        """Queue and immediately report whether ComfyUI accepted the graph.

        ComfyUI validates the whole graph before execution, so a successful
        /prompt means every node exists, every link type-checks and every model
        file named in a combo is actually present.
        """
        try:
            pid = self.queue(api_workflow)
            return True, pid
        except ComfyError as e:
            return False, str(e)

    def wait(self, prompt_id: str, timeout: float = 3600,
             poll: float = 1.0, sample_vram: bool = True) -> dict:
        """Block until the prompt finishes. Returns the history entry.

        Raises on execution error, including the node that failed.
        """
        peak = {"mb": 0}
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                try:
                    p = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5)
                    peak["mb"] = max(peak["mb"], int(p.stdout.strip().split("\n")[0]))
                except Exception:
                    pass
                stop.wait(0.5)

        t = None
        if sample_vram:
            t = threading.Thread(target=sampler, daemon=True)
            t.start()

        t0 = time.time()
        try:
            while True:
                if time.time() - t0 > timeout:
                    raise ComfyError(f"Timed out after {timeout}s waiting for {prompt_id}")
                try:
                    hist = self._get(f"/history/{prompt_id}")
                except Exception:
                    hist = {}
                if prompt_id in hist:
                    entry = hist[prompt_id]
                    status = entry.get("status", {})
                    if status.get("status_str") == "error" or \
                            any(m[0] == "execution_error" for m in status.get("messages", [])):
                        detail = json.dumps(status, indent=2)[:4000]
                        raise ComfyError(f"Execution failed for {prompt_id}:\n{detail}")
                    if status.get("completed") or entry.get("outputs"):
                        entry["_elapsed"] = time.time() - t0
                        entry["_peak_vram_mb"] = peak["mb"]
                        return entry
                time.sleep(poll)
        finally:
            stop.set()
            if t:
                t.join(timeout=2)

    def run(self, api_workflow: dict, timeout: float = 3600) -> dict:
        pid = self.queue(api_workflow)
        if self.log:
            self.log.info("queued prompt %s", pid)
        return self.wait(pid, timeout=timeout)

    @staticmethod
    def output_files(history_entry: dict, comfy_output: Path) -> list[Path]:
        """Absolute paths of everything the run saved."""
        files: list[Path] = []
        for _, out in (history_entry.get("outputs") or {}).items():
            for key in ("images", "videos", "gifs", "audio"):
                for item in out.get(key, []) or []:
                    sub = item.get("subfolder") or ""
                    fn = item.get("filename")
                    if fn:
                        files.append(comfy_output / sub / fn)
        return files


def load_api_workflow(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def set_input(wf: dict, class_type: str, input_name: str, value,
              title_contains: str | None = None, occurrence: int = 0) -> str:
    """Set an input on the Nth node of a given class (optionally by title).

    Returns the node id it changed. Raises if no such node exists, so a renamed
    node fails loudly instead of silently leaving a stale value in the graph.
    """
    hits = []
    for nid, nd in wf.items():
        if nd.get("class_type") != class_type:
            continue
        if title_contains and title_contains.lower() not in \
                (nd.get("_meta", {}).get("title", "")).lower():
            continue
        hits.append(nid)
    hits.sort(key=int)
    if len(hits) <= occurrence:
        raise KeyError(
            f"No node #{occurrence} of class {class_type}"
            + (f" with title containing {title_contains!r}" if title_contains else "")
            + f". Present classes: {sorted({n['class_type'] for n in wf.values()})}")
    wf[hits[occurrence]]["inputs"][input_name] = value
    return hits[occurrence]
