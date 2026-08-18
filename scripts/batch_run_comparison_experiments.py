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


BASELINE = "baseline_oneshot_mineru"
ABLATIONS = {"ablation_text_only", "ablation_no_l3", "ablation_large_chunk"}
DEFAULT_CONDITIONS = [BASELINE, "ablation_text_only", "ablation_no_l3", "ablation_large_chunk"]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def numeric_key(path: Path) -> tuple[int, str]:
    if path.name.isdigit():
        return int(path.name), path.name
    return 10**9, path.name


def split_conditions(values: list[str] | None) -> list[str]:
    if not values:
        return DEFAULT_CONDITIONS[:]
    result: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def read_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        path = Path(args.api_key_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


def discover_papers(root_dir: Path, start: int, end: int | None) -> list[Path]:
    papers = sorted([p for p in root_dir.iterdir() if p.is_dir()], key=numeric_key)
    if end is None:
        end = len(papers)
    return papers[max(start - 1, 0): min(end, len(papers))]


def build_env(project_dir: Path, api_key: str | None) -> dict[str, str]:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    entries = [str(project_dir), str(project_dir / "experiments")]
    if old_pythonpath:
        entries.append(old_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env["PAPER_MINING_ENGLISH_ONLY"] = "1"
    if api_key:
        env["DASHSCOPE_API_KEY"] = api_key
        env["QWEN_API_KEY"] = api_key
    return env


def stream_command(
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

        line_queue: queue.Queue[str | None] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                line_queue.put(line)
            line_queue.put(None)

        threading.Thread(target=read_stdout, daemon=True).start()

        start_time = time.time()
        last_heartbeat = start_time
        reader_done = False

        while process.poll() is None or not reader_done:
            try:
                line = line_queue.get(timeout=1)
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
                message = f"[{now()}] Timeout after {timeout}s; killed process.\n"
                print(message, end="", flush=True)
                log_file.write(message)
                return 124

            if heartbeat_seconds and time.time() - last_heartbeat >= heartbeat_seconds:
                log(f"Command still running ({int(elapsed)}s elapsed)")
                last_heartbeat = time.time()

    return int(process.returncode or 0)


def common_model_args(args: argparse.Namespace, api_key: str | None) -> list[str]:
    result: list[str] = []
    if args.base_url:
        result.extend(["--base-url", args.base_url])
    if args.model:
        result.extend(["--model", args.model])
    if api_key:
        result.extend(["--api-key", api_key])
    result.extend(["--request-timeout", str(args.request_timeout)])
    result.extend(["--max-tokens", str(args.max_tokens)])
    return result


def run_condition(
    args: argparse.Namespace,
    paper_dir: Path,
    project_dir: Path,
    condition: str,
    api_key: str | None,
) -> bool:
    paper_output = paper_dir / "outputs"
    batch_dir = paper_output / ".batch_experiments"
    marker = batch_dir / f"{condition}.done"
    log_path = batch_dir / "logs" / f"{condition}.log"

    if args.skip_existing and marker.exists():
        log(f"Skip existing {condition}: {marker}")
        return True

    if condition == BASELINE:
        output_dir = paper_output / "comparison_experiments" / "baseline_oneshot_mineru"
        cmd = [
            sys.executable,
            "-m",
            "experiments.one_shot_mineru_baseline",
            "--paper-dir",
            str(paper_dir),
            "--output-dir",
            str(output_dir),
            "--max-text-chars",
            str(args.oneshot_max_text_chars),
            "--max-images",
            str(args.oneshot_max_images),
        ]
        cmd.extend(common_model_args(args, api_key))
    elif condition in ABLATIONS:
        output_root = paper_output / "comparison_experiments" / "ablations"
        cmd = [
            sys.executable,
            "-m",
            "experiments.ablation_runner",
            "--paper-dir",
            str(paper_dir),
            "--output-root",
            str(output_root),
            "--conditions",
            condition,
            "--max-chars-per-chunk",
            str(args.max_chars_per_chunk),
            "--overlap-chars",
            str(args.overlap_chars),
            "--max-images-per-chunk",
            str(args.max_images_per_chunk),
            "--max-retries",
            str(args.max_retries),
            "--temperature",
            str(args.temperature),
            "--oneshot-max-text-chars",
            str(args.oneshot_max_text_chars),
            "--oneshot-max-images",
            str(args.oneshot_max_images),
            "--large-chunk-chars",
            str(args.large_chunk_chars),
            "--large-chunk-images",
            str(args.large_chunk_images),
        ]
        cmd.extend(common_model_args(args, api_key))
    else:
        log(f"Unknown condition: {condition}")
        return False

    log("Running: " + " ".join(cmd))
    code = stream_command(
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
        log(f"{condition} completed")
        return True

    log(f"{condition} failed with exit_code={code}; log={log_path}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-chars-per-chunk", type=int, default=9000)
    parser.add_argument("--overlap-chars", type=int, default=800)
    parser.add_argument("--max-images-per-chunk", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--oneshot-max-text-chars", type=int, default=60000)
    parser.add_argument("--oneshot-max-images", type=int, default=8)
    parser.add_argument("--large-chunk-chars", type=int, default=12000)
    parser.add_argument("--large-chunk-images", type=int, default=1)
    parser.add_argument("--condition-timeout", type=int, default=7200)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    project_dir = Path(args.project_dir).resolve()
    conditions = split_conditions(args.conditions)
    api_key = read_api_key(args)

    unknown = [item for item in conditions if item != BASELINE and item not in ABLATIONS]
    if unknown:
        log("Unknown conditions: " + ", ".join(unknown))
        return 2

    papers = discover_papers(root_dir, args.start, args.end)
    log(f"Discovered {len(papers)} paper folders under {root_dir}")
    log("English-only prompt constraint is enabled for comparison experiments")
    log("Conditions: " + ", ".join(conditions))

    failures: list[str] = []
    for paper_index, paper_dir in enumerate(papers, start=1):
        log(f"========== Paper {paper_index}/{len(papers)}: {paper_dir.name} ==========")
        for condition in conditions:
            ok = run_condition(args, paper_dir, project_dir, condition, api_key)
            if not ok:
                failures.append(f"{paper_dir.name}:{condition}")
                if args.stop_on_error:
                    log("Stopped on first error")
                    return 1
            if args.sleep:
                time.sleep(args.sleep)

    if failures:
        log("Failed jobs: " + ", ".join(failures))
        return 1

    log("All requested comparison experiments completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
