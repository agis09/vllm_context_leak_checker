# vLLM Cross-User Context Contamination Reproduction Tool

A tool for reproducing and automatically detecting cross-user context contamination that occurs when vLLM's async-scheduling is enabled. It simulates the same tool-call flow used by coding tools such as OpenCode and Cline, detecting contamination across concurrent multi-user requests.

## Setup

```bash
cd debug_tools
uv add openai
```

## Basic Usage

```bash
cd debug_tools
PYTHONUNBUFFERED=1 .venv/bin/python main.py \
    --base-url http://localhost:8000/v1 \
    --model moonshotai/Kimi-K2.5 \
    --concurrent-users 2 \
    --duration 1200 \
    --streaming \
    --stop-on-contamination
```

## User Configuration

N users are generated for each language (Japanese, Korean, Thai). With `--concurrent-users N`, N users per language are created, resulting in N×3 total concurrent users. Since the scripts differ completely between languages, cross-language contamination can be detected via script detection with zero false positives. Intra-language contamination is excluded from detection as it is difficult to distinguish from hallucination.

```
# With --concurrent-users 2 (6 users total)
[JP] user_000: 高橋雄一   | FALCON-5506
[JP] user_001: 中村さくら | EAGLE-3271
[KR] user_002: 김민준     | PHOENIX-7912
[KR] user_003: 이서연     | HAWK-4823
[TH] user_004: สมชาย      | SHADOW-1434
[TH] user_005: สุภาพร     | DRAGON-8019
```

## Two Execution Modes

### Round Mode (Default)

Runs N user sessions as one round, repeating for a specified number of rounds. Suitable for short verification runs.

```bash
.venv/bin/python main.py \
    --base-url http://localhost:8000/v1 \
    --model moonshotai/Kimi-K2.5 \
    --concurrent-users 2 \
    --rounds 10
```

### Sustained Load Mode (`--duration`)

N×3 workers continuously execute sessions for the specified number of seconds without interruption. Use this when applying sustained load for 5–20 minutes to reproduce the issue.

```bash
PYTHONUNBUFFERED=1 .venv/bin/python main.py \
    --base-url http://localhost:8000/v1 \
    --model moonshotai/Kimi-K2.5 \
    --concurrent-users 2 \
    --duration 1200 \
    --streaming \
    --stop-on-contamination \
    --seed 42
```

## Stop Conditions

| Condition | Behavior |
|---|---|
| `--duration` elapsed | Stops starting new sessions. In-progress sessions continue to completion |
| `--stop-on-contamination` | Immediately terminates the process (`os._exit`) when Layer 1 or 2 detects contamination |
| `--max-turns` reached | Terminates that session (default: 40) |
| Follow-ups exhausted | Session ends after all 15 follow-ups are consumed (see below) |
| Ctrl+C | Stops the process |

When `--stop-on-contamination` is used, the process writes a context dump at the moment of detection and then terminates immediately, without waiting for in-flight API requests to complete.

**About follow-up consumption:** In each session, after the assistant's final response, a new task is injected as a "follow-up" user message to continue the conversation. This mimics real coding tool usage (moving on to the next task after finishing a bug investigation). Up to 15 follow-ups can be injected per session. Once all 15 are consumed, the session ends (even if the `--max-turns` limit has not been reached).

## Detection Mechanism

### 3-Layer Detection

#### Layer 1: Canary Tokens (Real-time, Zero False Positives)

Each user is assigned 7 unique tokens, and the system checks whether they appear in other users' responses.

| Token | Example |
|---|---|
| `canary_uuid` | `CANARY-63ef16aa-8274-477c-87eb-7989d6b39943` |
| `canary_build` | `BUILD-52769f015610` |
| `persona_name` | `中村さくら` |
| `project_code` | `FALCON-5506` |
| `unique_class` | `NovaEngine328` |
| `unique_var` | `ctx_a1b2c3d4` |
| `unique_const` | `MAX_RETRY_4582` |

If any of these strings appear in the `content` or `reasoning` of a response, contamination is detected. All directions (JP↔KR↔TH) are covered.

#### Layer 2: Script Detection (Real-time)

Detects the presence of characters from a script that does not match the expected language of the session.

| Session | Detection Target | Notes |
|---|---|---|
| Japanese (JP) | Thai script | Hangul excluded as it rarely appears via hallucination |
| Korean (KR) | Hiragana/Katakana, Thai script | |
| Thai (TH) | Hiragana/Katakana, Hangul | |

Even a single character triggers a contamination flag.

#### Layer 3: N-gram Similarity (Post-session, Batch)

Calculates character 5-gram similarity between responses from different users. If similarity exceeds the threshold (default: 30%), it is flagged. This runs as part of the final analysis after session completion.

### Tool-Call Flow Reproduction

Reproduces the same flow as real coding tools:

```
1. [system]    System prompt with canary tokens (language-specific)
2. [user]      "Investigate this bug"
3. [assistant] → tool_call: read_file("src/main.py")
4. [tool]      Large pseudo-code with canary tokens (~150 lines)
5. [assistant] → tool_call: search_code("bug")
6. [tool]      Search results with canary tokens
7. [assistant] Final response → follow-up injection moves to next task
...(continues for 30+ turns)
```

### Sliding Window

When the context exceeds `--context-target` (default: 120,000 tokens), older messages are removed while preserving the system prompt and initial request. This keeps each request stable around 100K+ tokens, maintaining sustained large-context load on the server.

## Options

| Option | Default | Description |
|---|---|---|
| `--base-url` | (required) | vLLM OpenAI-compatible API endpoint |
| `--model` | (required) | Model name (e.g., `moonshotai/Kimi-K2.5`) |
| `--concurrent-users` | 2 | Concurrent users per language (total = N × 3 languages) |
| `--rounds` | 10 | Number of test rounds (round mode) |
| `--duration` | none | Duration in seconds for sustained load (enables sustained load mode) |
| `--max-turns` | 40 | Maximum turns per session |
| `--streaming` | off | Use streaming API |
| `--timeout` | 300 | Request timeout (seconds) |
| `--output` | auto-generated | Output path for JSON report |
| `--output-dir` | `./experiments/<timestamp>/` | Experiment output directory |
| `--extra-body` | none | Additional JSON to include in requests |
| `--ngram-threshold` | 0.30 | N-gram similarity threshold |
| `--seed` | none | Random seed for profile generation |
| `--stop-on-contamination` | off | Immediately terminate the process on Layer 1/2 detection |
| `--context-target` | 120000 | Target token count for the sliding window |

## Output

### Experiment Directory

Each run outputs to `experiments/<timestamp>/` (configurable with `--output-dir`).

```
experiments/20260404_160000/
├── report.json                # Final report (config, profiles, detection results, summary)
├── turns.jsonl                # Per-turn metrics (timestamp, context length, latency)
├── sessions.jsonl             # Full content of all turns (content, reasoning, tool_calls)
└── contamination_1.json       # Context dump on detection (includes surrounding turns)
```

### Turn Log (`turns.jsonl`)

Each turn is recorded as a single JSON line. Contains data for post-hoc analysis of request load and response speed.

| Field | Type | Description |
|---|---|---|
| `ts` | float | Elapsed seconds since experiment start |
| `user_id` | string | User ID (e.g., `user_000`) |
| `persona` | string | Persona name |
| `language` | string | Session language (`japanese`, `korean`, `thai`) |
| `turn` | int | Turn number within the session |
| `content_len` | int | Character count of response `content` |
| `reasoning_len` | int | Character count of response `reasoning` |
| `has_tool_calls` | bool | Whether the response includes tool calls |
| `context_chars` | int | Total character count of all messages at request time |
| `context_tokens_est` | int | Estimated token count at request time (~3.5 chars/token) |
| `message_count` | int | Number of messages at request time |
| `request_duration_s` | float | Seconds from API request to response completion |
| `response_chars` | int | Total response character count (content + reasoning) |

#### Analysis Example

Example of aggregating average latency and throughput by time bucket:

```python
import json
lines = [json.loads(l) for l in open("turns.jsonl")]

# Average latency per 30-second bucket
from collections import defaultdict
buckets = defaultdict(list)
for t in lines:
    bucket = int(t["ts"] // 30) * 30
    buckets[bucket].append(t)

for sec in sorted(buckets):
    entries = buckets[sec]
    avg_lat = sum(e["request_duration_s"] for e in entries) / len(entries)
    avg_chars_per_s = sum(
        e["response_chars"] / e["request_duration_s"]
        for e in entries if e["request_duration_s"] > 0
    ) / len(entries)
    print(f"  {sec:>5}s: avg_latency={avg_lat:.1f}s  avg_throughput={avg_chars_per_s:.0f} chars/s  n={len(entries)}")
```

### Context Dump (`contamination_N.json`)

Automatically saved when contamination is detected. Contains the full response of the detection turn and the contents of the preceding 5 turns.

### Console Output

Progress is displayed every 30 seconds:

```
  [120s / 1200s] turns: 22 | sessions: 0 done | contaminations: 0 | errors: 0 | 1080s left
    workers: user_000:t4 | user_001:t9 | user_002:t5

  !! CONTAMINATION DETECTED in user_001 (문서진) turn 13: 1 finding(s) !!
    [LANG]  user_001 | expected=korean detected=japanese turn=13 field=content count=16
  >> Context dump saved: experiments/20260404_160000/contamination_1.json

  *** Contamination detected — stopping immediately (--stop-on-contamination) ***
```
