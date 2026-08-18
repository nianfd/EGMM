"""Run only the large-chunk ablation for a folder of MinerU-parsed papers.

This helper intentionally runs a single condition, ``ablation_large_chunk``.
It keeps outputs under each paper folder so previously completed baseline and
ablation results are not touched.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


CONDITION = "ablation_large_chunk"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def numeric_key(path: Path) -> tuple[int, str]:
    return (int(path.name), path.name) if path.name.isdigit() else (10**9, path.name)


def read_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        key_path = Path(args.api_key_file)
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()
    return None


def discover_papers(root_dir: Path, start: int, end: int | None) -> list[Path]:
    papers = sorted([p for p in root_dir.iterdir() if p.is_dir()], key=numeric_key)
    if end is None:
        end = len(papers)
    return papers[max(start - 1, 0) : min(end, len(papers))]


def stream_process(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: int,
    heartbeat_seconds: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} =====\n")
        log_file.write("COMMAND: " + " ".join(cmd) + "\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        output_queue: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)
            output_queue.put(None)

        threading.Thread(target=reader, daemon=True).start()

        start_time = time.time()
        last_heartbeat = start_time
        reader_done = False

        while process.poll() is None or not reader_done:
            try:
                line = output_queue.get(timeout=1)
                if line is None:
                    reader_done = True
                    continue
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            except queue.Empty:
                pass

            elapsed = time.time() - start_time
            if timeout and elapsed > timeout and process.poll() is None:
                process.kill()
                msg = f"[{now()}] Timeout after {timeout}s; killed process.\n"
                print(msg, end="", flush=True)
                log_file.write(msg)
                return 124

            if heartbeat_seconds and time.time() - last_heartbeat >= heartbeat_seconds:
                log(f"{CONDITION} still running ({int(elapsed)}s elapsed)")
                last_heartbeat = time.time()

        return int(process.returncode or 0)


def build_env(project_dir: Path, api_key: str | None) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    entries = [str(project_dir), str(project_dir / "experiments")]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    if api_key:
        env["DASHSCOPE_API_KEY"] = api_key
        env["QWEN_API_KEY"] = api_key
    env["PAPER_MINING_ENGLISH_ONLY"] = "1"
    return env


def run_paper(args: argparse.Namespace, paper_dir: Path, index: int, total: int, api_key: str | None) -> bool:
    project_dir = Path(args.project_dir).resolve()
    paper_output = paper_dir / "outputs"
    output_root = paper_output / "comparison_experiments" / "ablations"
    batch_dir = paper_output / ".batch_experiments"
    marker = batch_dir / f"{CONDITION}.done"
    log_path = batch_dir / "logs" / f"{CONDITION}.log"

    log(f"========== Paper {index}/{total}: {paper_dir.name} ==========")
    if args.skip_existing and marker.exists():
        log(f"Skip existing {CONDITION}: {marker}")
        return True

    cmd = [
        sys.executable,
        "-m",
        "experiments.ablation_runner",
        "--paper-dir",
        str(paper_dir),
        "--output-root",
        str(output_root),
        "--conditions",
        CONDITION,
        "--large-chunk-chars",
        str(args.large_chunk_chars),
        "--large-chunk-images",
        str(args.large_chunk_images),
        "--max-tokens",
        str(args.max_tokens),
        "--request-timeout",
        str(args.request_timeout),
        "--max-retries",
        str(args.max_retries),
        "--temperature",
        str(args.temperature),
    ]
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.model:
        cmd.extend(["--model", args.model])
    if api_key:
        cmd.extend(["--api-key", api_key])

    log("Running: " + " ".join(cmd))
    code = stream_process(
        cmd=cmd,
        cwd=project_dir,
        env=build_env(project_dir, api_key),
        log_path=log_path,
        timeout=args.condition_timeout,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    if code == 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
        log(f"{CONDITION} completed -> {output_root}")
        return True

    log(f"{CONDITION} failed with exit_code={code}; log={log_path}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--conditions", nargs="*", default=[CONDITION])
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--large-chunk-chars", type=int, default=12000)
    parser.add_argument("--large-chunk-images", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--condition-timeout", type=int, default=7200)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = [item for value in args.conditions for item in value.split(",") if item]
    if requested != [CONDITION]:
        raise SystemExit("This helper only supports --conditions ablation_large_chunk")
    root_dir = Path(args.root_dir).resolve()
    papers = discover_papers(root_dir, args.start, args.end)
    api_key = read_api_key(args)

    log(f"Discovered {len(papers)} paper folders under {root_dir}")
    log("Only ablation_large_chunk will be executed")

    failures: list[str] = []
    for idx, paper_dir in enumerate(papers, start=1):
        ok = run_paper(args, paper_dir, idx, len(papers), api_key)
        if not ok:
            failures.append(paper_dir.name)
            if args.stop_on_error:
                break
        if args.sleep:
            time.sleep(args.sleep)

    if failures:
        log("Failed papers: " + ", ".join(failures))
        return 1
    log("All requested large-chunk ablations completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
