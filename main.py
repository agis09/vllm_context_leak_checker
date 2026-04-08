"""
vLLM Context Contamination Reproduction Tool

Simulates multiple concurrent coding-tool sessions (like OpenCode / Cline)
against a vLLM endpoint and automatically detects cross-user context
contamination via canary tokens, language-script analysis, and n-gram
similarity.

Usage:
    # 2 users per language (6 total: 2 JP + 2 KR + 2 TH):
    python debug_tools/main.py \
        --base-url http://localhost:8000/v1 \
        --model moonshotai/Kimi-K2.5 \
        --concurrent-users 2 \
        --rounds 10

    # With streaming (closer to real coding-tool behavior):
    python debug_tools/main.py \
        --base-url http://localhost:8000/v1 \
        --model moonshotai/Kimi-K2.5 \
        --concurrent-users 3 \
        --rounds 20 \
        --streaming

    # With thinking enabled (for Kimi K2.5):
    python debug_tools/main.py \
        --base-url http://localhost:8000/v1 \
        --model moonshotai/Kimi-K2.5 \
        --concurrent-users 2 \
        --rounds 10 \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":true}}'
"""

import argparse
import asyncio
import json
import sys
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

# Ensure imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user_profile import generate_profiles, UserProfile
from session import simulate_session
from detector import analyze_all_sessions, analyze_session, detect_turn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="vLLM Context Contamination Reproduction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--base-url", required=True,
        help="vLLM OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)",
    )
    p.add_argument(
        "--model", required=True,
        help="Model name served by vLLM (e.g. moonshotai/Kimi-K2.5)",
    )
    p.add_argument(
        "--concurrent-users", type=int, default=2,
        help="Concurrent simulated users per language (default: 2, total = N * 3 languages)",
    )
    p.add_argument(
        "--rounds", type=int, default=10,
        help="Number of test rounds (default: 10)",
    )
    p.add_argument(
        "--max-turns", type=int, default=40,
        help="Max API calls per session (default: 40, enables 30+ turn sessions)",
    )
    p.add_argument(
        "--streaming", action="store_true",
        help="Use streaming API (closer to real coding-tool behavior)",
    )
    p.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output JSON report path (default: auto-generated with timestamp)",
    )
    p.add_argument(
        "--extra-body", default=None,
        help=(
            "Extra request body as JSON string, e.g. "
            '\'{"chat_template_kwargs":{"enable_thinking":true}}\''
        ),
    )
    p.add_argument(
        "--ngram-threshold", type=float, default=0.30,
        help="N-gram similarity threshold for flagging (default: 0.30)",
    )
    p.add_argument(
        "--duration", type=int, default=None,
        help=(
            "Sustained load duration in seconds (e.g. 600 for 10 minutes). "
            "When set, workers continuously run sessions for the specified "
            "duration instead of using rounds. This keeps the server under "
            "constant load without gaps between rounds."
        ),
    )
    p.add_argument(
        "--stop-on-contamination", action="store_true",
        help="Stop immediately when token/language contamination is detected",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible profile generation",
    )
    p.add_argument(
        "--context-target", type=int, default=120_000,
        help=(
            "Target context size in tokens (default: 120000). "
            "Messages are trimmed via sliding window to stay near this size. "
            "Prevents unbounded context growth that slows each request."
        ),
    )
    p.add_argument(
        "--output-dir", default=None,
        help=(
            "Directory to store all experiment outputs (report, turn log, "
            "contamination dumps). A timestamped subdirectory is created "
            "automatically. Default: ./experiments/<timestamp>/"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

LANG_TAGS = {"japanese": "JP", "korean": "KR", "thai": "TH"}


def print_profiles(profiles: list[UserProfile]):
    print(f"\n{'=' * 72}")
    print(f"  User Profiles ({len(profiles)} users)")
    print(f"{'=' * 72}")
    for p in profiles:
        tag = LANG_TAGS.get(p.language, "??")
        print(f"  [{tag}] {p.user_id}: {p.persona_name} | {p.project_code} | {p.unique_class}")
    print()


def print_round_header(round_num: int, total: int, n_users: int):
    print(f"\n{'─' * 72}")
    print(f"  Round {round_num + 1}/{total} — {n_users} concurrent users")
    print(f"{'─' * 72}")


def print_session_status(session: dict):
    uid = session["user_id"]
    persona = session["persona"]
    turns = session["total_turns"]
    err = session.get("error")
    if err:
        print(f"  {uid} ({persona}): ERROR - {err[:80]}")
    else:
        tool_turns = sum(1 for t in session["turns"] if t.get("tool_calls"))
        print(f"  {uid} ({persona}): {turns} turns ({tool_turns} with tool calls)")


def print_findings(findings: list[dict], round_num: int | None = None, label: str | None = None):
    tag = label or (f"Round {round_num + 1}" if round_num is not None else "session")
    if not findings:
        print(f"  >> {tag}: No contamination detected")
        return

    print(f"\n  !! CONTAMINATION DETECTED in {tag}: "
          f"{len(findings)} finding(s) !!")
    for f in findings:
        if f["type"] == "token":
            print(
                f"    [TOKEN] {f['source_user']} ({f.get('source_persona', '?')}) "
                f"-> {f['victim_user']} | "
                f"turn={f['turn']} field={f['field']} "
                f"after_tool={f['is_after_tool_result']} | "
                f"{f['token_type']}={f['token_value'][:50]}..."
            )
        elif f["type"] == "language":
            print(
                f"    [LANG]  {f['victim_user']} | "
                f"expected={f['expected']} detected={f['detected']} "
                f"turn={f['turn']} field={f['field']} "
                f"count={f['count']}"
            )
        elif f["type"] == "ngram_similarity":
            print(
                f"    [NGRAM] {f['user_a']}({f['field_a']}) <-> "
                f"{f['user_b']}({f['field_b']}) | "
                f"similarity={f['similarity']:.2%} "
                f"shared={f['shared_ngrams']}"
            )


def _dump_contamination_context(
    profile: "UserProfile",
    turn_data: dict,
    turn_history: dict[str, list[dict]],
    output_base: str,
    finding_num: int,
):
    """Write contaminated turn + surrounding context to a dedicated file."""
    report_dir = os.path.dirname(output_base) or "."
    dump_path = os.path.join(report_dir, f"contamination_{finding_num}.json")
    # Current (contaminated) turn
    contaminated = {
        "user_id": profile.user_id,
        "persona": profile.persona_name,
        "language": profile.language,
        "turn": turn_data["turn"],
        "content": turn_data.get("content"),
        "reasoning": turn_data.get("reasoning"),
        "has_tool_calls": bool(turn_data.get("tool_calls")),
    }
    # Preceding turns from same worker
    preceding = list(turn_history.get(profile.user_id, []))
    # Remove the last entry if it's the same turn (already in contaminated)
    if preceding and preceding[-1]["turn"] == turn_data["turn"]:
        preceding = preceding[:-1]

    dump = {
        "finding_num": finding_num,
        "contaminated_turn": contaminated,
        "preceding_turns": preceding,
    }
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2, default=str)

    print(f"  >> Context dump saved: {dump_path}")

    # Also print a snippet to stdout for immediate visibility
    print(f"\n  --- Contaminated response ({profile.user_id}, turn {turn_data['turn']}) ---")
    for field in ("content", "reasoning"):
        text = turn_data.get(field)
        if text:
            # Show first 500 and last 500 chars
            if len(text) > 1200:
                snippet = text[:500] + "\n  [...truncated...]\n" + text[-500:]
            else:
                snippet = text
            print(f"  [{field}] ({len(text)} chars):")
            for line in snippet.split("\n"):
                print(f"    {line}")
    if preceding:
        prev = preceding[-1]
        print(f"\n  --- Previous turn (turn {prev['turn']}) ---")
        for field in ("content", "reasoning"):
            text = prev.get(field)
            if text:
                if len(text) > 600:
                    snippet = text[:300] + "\n  [...truncated...]\n" + text[-300:]
                else:
                    snippet = text
                print(f"  [{field}] ({len(text)} chars):")
                for line in snippet.split("\n"):
                    print(f"    {line}")
    print(f"  {'─' * 60}\n")


def print_final_summary(report: dict, elapsed: float):
    s = report["summary"]
    print(f"\n{'=' * 72}")
    print(f"  FINAL REPORT")
    print(f"{'=' * 72}")
    print(f"  Total sessions:               {s['total_sessions']}")
    print(f"  Sessions with errors:         {s['sessions_with_errors']}")
    print(f"  Total contamination findings: {s['total_findings']}")
    print(f"    Token contaminations:       {s['token_contaminations']}")
    print(f"    Language contaminations:     {s['language_contaminations']}")
    print(f"    N-gram similarity alerts:   {s['ngram_similarity_alerts']}")
    print(f"    After tool result:          {s['contaminations_after_tool_result']}")
    print(f"    Before tool result:         {s['contaminations_before_tool_result']}")

    if s["turn_distribution"]:
        print(f"\n  Turn distribution:")
        for turn, count in sorted(s["turn_distribution"].items(), key=lambda x: int(x[0])):
            bar = "#" * min(count, 40)
            print(f"    Turn {turn}: {count:>4}  {bar}")

    if s["field_distribution"]:
        print(f"\n  Field distribution:")
        for field, count in s["field_distribution"].items():
            print(f"    {field}: {count}")

    if s["contamination_matrix"]:
        print(f"\n  Contamination matrix (source -> victim):")
        for src, victims in s["contamination_matrix"].items():
            for victim, count in victims.items():
                print(f"    {src} -> {victim}: {count} time(s)")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    if s["total_findings"] == 0:
        print(f"\n  RESULT: No contamination detected.")
    else:
        print(f"\n  RESULT: CONTAMINATION DETECTED "
              f"— {s['total_findings']} finding(s) across sessions.")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Main — round mode
# ---------------------------------------------------------------------------

async def run_rounds(args, profiles, extra_body):
    """Round-based mode: launch N concurrent sessions per round."""
    all_sessions: list[dict] = []
    all_findings: list[dict] = []
    turn_count = 0
    contamination_count = 0
    lock = asyncio.Lock()
    start_time = time.time()

    async def on_turn_complete(profile: UserProfile, turn_data: dict):
        nonlocal turn_count, contamination_count
        async with lock:
            turn_count += 1
            findings = detect_turn(
                profile.user_id, profile.language, turn_data, profiles,
            )
            if findings:
                for f in findings:
                    f["elapsed_sec"] = round(time.time() - start_time, 1)
                all_findings.extend(findings)
                contamination_count += len(findings)
                print_findings(
                    findings,
                    label=(
                        f"{profile.user_id} ({profile.persona_name}) "
                        f"turn {turn_data['turn']}"
                    ),
                )

    for round_num in range(args.rounds):
        print_round_header(round_num, args.rounds, len(profiles))

        tasks = [
            simulate_session(
                profile=profile,
                base_url=args.base_url,
                model=args.model,
                max_turns=args.max_turns,
                streaming=args.streaming,
                timeout=args.timeout,
                extra_body=extra_body,
                on_turn_complete=on_turn_complete,
                context_target_tokens=args.context_target,
            )
            for profile in profiles
        ]
        round_sessions = await asyncio.gather(*tasks)

        for session in round_sessions:
            print_session_status(session)

        for session in round_sessions:
            session["round"] = round_num
        all_sessions.extend(round_sessions)

        round_report = analyze_all_sessions(
            list(round_sessions), profiles,
            ngram_threshold=args.ngram_threshold,
        )
        round_findings = round_report["findings"]
        for f in round_findings:
            f["round"] = round_num
        all_findings.extend(round_findings)

        print_findings(round_findings, round_num=round_num)

    return all_sessions, all_findings, time.time() - start_time


# ---------------------------------------------------------------------------
# Main — sustained duration mode
# ---------------------------------------------------------------------------

async def run_continuous(args, profiles, extra_body):
    """Sustained load mode: N workers continuously run sessions for --duration
    seconds without any gaps, keeping the server under constant pressure.

    Each worker is assigned a fixed user profile and runs sessions
    back-to-back. Multiple workers overlap so the server always has
    concurrent in-flight requests at different stages of the tool-call loop.

    Detection runs on every turn in real-time via on_turn_complete callback.
    """
    duration = args.duration
    end_time = time.time() + duration
    start_time = time.time()

    all_sessions: list[dict] = []
    all_findings: list[dict] = []
    lock = asyncio.Lock()
    stop_event = asyncio.Event()
    stop_on_contamination = args.stop_on_contamination

    # Shared stats (updated under lock)
    stats = {
        "turns": 0,
        "sessions": 0,
        "contaminations": 0,
        "errors": 0,
        "done": False,
        # Per-worker current turn (worker_idx -> turn_idx)
        "worker_turns": {p.user_id: 0 for p in profiles},
    }

    # Per-worker recent turn history for context around contamination
    turn_history: dict[str, list[dict]] = defaultdict(list)
    HISTORY_WINDOW = 5  # keep last N turns per worker

    # JSONL log for incremental persistence (survives kill)
    output_base = args.output
    # Place turns log in same directory as the report
    report_dir = os.path.dirname(output_base) or "."
    turns_log_path = os.path.join(report_dir, "turns.jsonl")
    turns_log_file = open(turns_log_path, "w", encoding="utf-8")
    sessions_log_path = os.path.join(report_dir, "sessions.jsonl")
    sessions_log_file = open(sessions_log_path, "w", encoding="utf-8")

    async def on_turn_complete(profile: UserProfile, turn_data: dict):
        """Called after every API response — runs detection immediately."""
        async with lock:
            stats["turns"] += 1
            stats["worker_turns"][profile.user_id] = turn_data["turn"]

            # Save turn to history ring buffer
            history = turn_history[profile.user_id]
            history.append({
                "turn": turn_data["turn"],
                "content": turn_data.get("content"),
                "reasoning": turn_data.get("reasoning"),
                "has_tool_calls": bool(turn_data.get("tool_calls")),
            })
            if len(history) > HISTORY_WINDOW:
                history.pop(0)

            # Write every turn to JSONL for persistence
            log_entry = {
                "ts": round(time.time() - start_time, 1),
                "user_id": profile.user_id,
                "persona": profile.persona_name,
                "language": profile.language,
                "turn": turn_data["turn"],
                "content_len": len(turn_data.get("content") or ""),
                "reasoning_len": len(turn_data.get("reasoning") or ""),
                "has_tool_calls": bool(turn_data.get("tool_calls")),
                "context_chars": turn_data.get("context_chars", 0),
                "context_tokens_est": turn_data.get("context_tokens_est", 0),
                "message_count": turn_data.get("message_count", 0),
                "request_duration_s": turn_data.get("request_duration_s", 0),
                "response_chars": turn_data.get("response_chars", 0),
            }
            turns_log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            turns_log_file.flush()

            # Full session content log
            session_entry = {
                "ts": log_entry["ts"],
                "user_id": profile.user_id,
                "persona": profile.persona_name,
                "language": profile.language,
                "turn": turn_data["turn"],
                "content": turn_data.get("content"),
                "reasoning": turn_data.get("reasoning"),
                "tool_calls": turn_data.get("tool_calls"),
                "finish_reason": turn_data.get("finish_reason"),
                "context_chars": turn_data.get("context_chars", 0),
                "context_tokens_est": turn_data.get("context_tokens_est", 0),
                "message_count": turn_data.get("message_count", 0),
                "request_duration_s": turn_data.get("request_duration_s", 0),
                "response_chars": turn_data.get("response_chars", 0),
            }
            sessions_log_file.write(json.dumps(session_entry, ensure_ascii=False) + "\n")
            sessions_log_file.flush()

            findings = detect_turn(
                profile.user_id, profile.language, turn_data, profiles,
            )
            if findings:
                elapsed = time.time() - start_time
                for f in findings:
                    f["elapsed_sec"] = round(elapsed, 1)
                    f["global_turn"] = stats["turns"]
                all_findings.extend(findings)
                stats["contaminations"] += len(findings)
                print_findings(
                    findings,
                    label=(
                        f"{profile.user_id} ({profile.persona_name}) "
                        f"turn {turn_data['turn']}"
                    ),
                )

                # Dump context around contamination
                _dump_contamination_context(
                    profile, turn_data, turn_history, output_base,
                    stats["contaminations"],
                )

                has_token_hit = any(
                    f["type"] == "token" for f in findings
                )
                if stop_on_contamination and has_token_hit:
                    print(
                        f"\n  *** Token contamination (Layer 1) detected "
                        f"— stopping immediately "
                        f"(--stop-on-contamination) ***"
                    )
                    sys.stdout.flush()
                    turns_log_file.close()
                    sessions_log_file.close()
                    os._exit(0)

    async def worker(profile: UserProfile):
        while time.time() < end_time and not stop_event.is_set():
            # Reset turn counter for this worker at session start
            async with lock:
                stats["worker_turns"][profile.user_id] = 0

            session = await simulate_session(
                profile=profile,
                base_url=args.base_url,
                model=args.model,
                max_turns=args.max_turns,
                streaming=args.streaming,
                timeout=args.timeout,
                extra_body=extra_body,
                on_turn_complete=on_turn_complete,
                context_target_tokens=args.context_target,
                stop_event=stop_event,
            )

            async with lock:
                stats["sessions"] += 1
                session["session_num"] = stats["sessions"]
                all_sessions.append(session)
                if session.get("error"):
                    stats["errors"] += 1
                    elapsed = time.time() - start_time
                    err_msg = session["error"][:120]
                    print(
                        f"\n  [ERROR] {profile.user_id} ({profile.persona_name}) "
                        f"at {elapsed:.0f}s, turn {session['total_turns']}: "
                        f"{err_msg}"
                    )

    async def status_printer():
        """Print periodic status every 30 seconds."""
        while not stats["done"]:
            await asyncio.sleep(30)
            if stats["done"]:
                break
            elapsed = time.time() - start_time
            remaining = max(0, end_time - time.time())
            worker_info = " | ".join(
                f"{uid}:t{t}" for uid, t in stats["worker_turns"].items()
            )
            print(
                f"\n  [{elapsed:.0f}s / {duration}s] "
                f"turns: {stats['turns']} | "
                f"sessions: {stats['sessions']} done | "
                f"contaminations: {stats['contaminations']} | "
                f"errors: {stats['errors']} | "
                f"{remaining:.0f}s left"
            )
            print(f"    workers: {worker_info}")

    print(f"\n  Starting sustained load: {len(profiles)} workers x "
          f"{duration}s ({duration // 60}m{duration % 60:02d}s)")
    print(f"  Real-time detection: ON (per-turn)")
    print(f"  Press Ctrl+C to stop early\n")

    async def deadline_enforcer():
        """Force-cancel all workers when duration expires."""
        await asyncio.sleep(duration + 30)  # grace period of 30s after duration
        if not stats["done"]:
            print("\n  *** Duration exceeded — force-stopping all workers ***")
            stop_event.set()

    status_task = asyncio.create_task(status_printer())
    deadline_task = asyncio.create_task(deadline_enforcer())
    worker_tasks = [asyncio.create_task(worker(profile)) for profile in profiles]
    try:
        await asyncio.gather(*worker_tasks)
    except asyncio.CancelledError:
        pass
    finally:
        stats["done"] = True
        status_task.cancel()
        deadline_task.cancel()
        # Cancel any remaining worker tasks
        for t in worker_tasks:
            if not t.done():
                t.cancel()
        try:
            await status_task
        except asyncio.CancelledError:
            pass

    turns_log_file.close()
    sessions_log_file.close()
    print(f"  Turn log saved: {turns_log_path}")
    print(f"  Session log saved: {sessions_log_path}")

    elapsed = time.time() - start_time
    print(
        f"\n  Sustained load finished: "
        f"{stats['sessions']} sessions, {stats['turns']} turns in {elapsed:.1f}s"
    )

    return all_sessions, all_findings, elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def setup_experiment_dir(args) -> str:
    """Create experiment output directory and return its path.

    Directory structure:
        <output_dir>/<timestamp>/
            report.json
            turns.jsonl
            contamination_1.json
            ...
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        exp_dir = os.path.join(args.output_dir, ts)
    else:
        exp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "experiments", ts,
        )
    os.makedirs(exp_dir, exist_ok=True)

    # Resolve output paths if not explicitly set
    if not args.output:
        args.output = os.path.join(exp_dir, "report.json")
    return exp_dir


async def run(args):
    extra_body = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            print(f"Error: --extra-body is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    exp_dir = setup_experiment_dir(args)
    print(f"\n  Experiment dir: {exp_dir}")

    profiles = generate_profiles(args.concurrent_users, seed=args.seed)
    print_profiles(profiles)

    mode = "duration" if args.duration else "rounds"
    if mode == "duration":
        print(f"  Mode:      sustained ({args.duration}s)")
    else:
        print(f"  Mode:      rounds ({args.rounds} rounds)")
    total_users = len(profiles)
    print(f"  Workers:   {total_users} concurrent users ({args.concurrent_users} per language x 3 languages)")
    print(f"  Max turns: {args.max_turns} per session")
    print(f"  Model:     {args.model}")
    print(f"  Endpoint:  {args.base_url}")
    print(f"  Streaming: {args.streaming}")
    print(f"  Context target: ~{args.context_target:,} tokens (sliding window)")
    if extra_body:
        print(f"  Extra body: {json.dumps(extra_body)}")

    if mode == "duration":
        all_sessions, all_findings, elapsed = await run_continuous(
            args, profiles, extra_body,
        )
    else:
        all_sessions, all_findings, elapsed = await run_rounds(
            args, profiles, extra_body,
        )

    # Final aggregated analysis (includes n-gram cross-session comparison)
    final_report = analyze_all_sessions(
        all_sessions, profiles,
        ngram_threshold=args.ngram_threshold,
    )
    # Merge real-time findings (layer 1/2) with batch findings (layer 3 n-gram)
    ngram_findings = [f for f in final_report["findings"] if f["type"] == "ngram_similarity"]
    final_report["findings"] = all_findings + ngram_findings
    final_report["config"] = {
        "mode": mode,
        "base_url": args.base_url,
        "model": args.model,
        "users_per_language": args.concurrent_users,
        "total_users": len(profiles),
        "rounds": args.rounds,
        "duration": args.duration,
        "max_turns": args.max_turns,
        "streaming": args.streaming,
        "timeout": args.timeout,
        "extra_body": extra_body,
        "ngram_threshold": args.ngram_threshold,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    final_report["profiles"] = [
        {
            "user_id": p.user_id,
            "language": p.language,
            "persona_name": p.persona_name,
            "project_code": p.project_code,
            "canary_uuid": p.canary_uuid,
            "canary_build": p.canary_build,
            "unique_class": p.unique_class,
            "unique_var": p.unique_var,
            "unique_const": p.unique_const,
        }
        for p in profiles
    ]

    print_final_summary(final_report, elapsed)

    output_path = args.output or (
        f"contamination_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Report saved: {output_path}\n")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
