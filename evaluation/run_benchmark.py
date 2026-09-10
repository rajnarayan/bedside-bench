#!/usr/bin/env python3
"""Generate, judge, score, resume, and summarize a BedsideBench evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from evaluation.f1_weighted import compute_f1_weighted
except ImportError:  # script invocation: python3 evaluation/run_benchmark.py
    from f1_weighted import compute_f1_weighted

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "bedside-bench-grading-v1"
ANSWER_MAX_TOKENS = 32000
# Answer models sample at temperature 1 (provider default sampling); the judge
# stays at 0. Models that reject an explicit temperature fall back to omitting
# it and record null.
ANSWER_TEMPERATURE = 1
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 5000
API_URL = "https://api.openai.com/v1/chat/completions"
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

ANSWER_SYSTEM = """You are a physician answering a clinical question. Give a direct, self-contained answer that addresses the requested evaluation or management. Explain essential reasoning, calculations, safety precautions, and follow-up. Do not mention benchmarks, rubrics, or grading."""

JUDGE_SYSTEM = """You are grading a clinical answer against independent rubric items.

For every rubric item, decide whether the ANSWER explicitly states or clearly entails it.

Rules:
1. Judge only information present in the ANSWER. Do not infer omitted content.
2. Accept clinically equivalent wording; exact phrase matching is not required.
3. A partially stated criterion is false when a clinically essential component is missing.
4. For a harmful, inappropriate, or negative-point item, true means the ANSWER endorses, recommends, or performs that action. Merely mentioning the action to reject or warn against it is false.
5. Judge each item independently.
6. Return exactly one decision for every supplied item_id and no other item_ids.

Return JSON only:
{"decisions":[{"item_id":"item_0","matched":true},{"item_id":"item_1","matched":false}]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--judge-model", default="gpt-5.6-sol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Answer-model reasoning. Use 'max' for the highest setting each provider accepts.",
    )
    return parser.parse_args()


def resolve_reasoning_effort(model: str, mode: str | None) -> str | None:
    """Map a requested effort, including 'max', onto the provider's highest supported value."""
    if not mode:
        return None
    if mode != "max":
        return mode
    if is_gemini_model(model):
        return "high"
    if is_anthropic_model(model):
        return "max"
    if is_fireworks_model(model):
        return "max"
    return "xhigh"


def load_token() -> str:
    for name in ("OPENAI_ACCESS_TOKEN", "OPENAI_API_KEY"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    raise SystemExit("Set OPENAI_ACCESS_TOKEN or OPENAI_API_KEY")


def load_fireworks_token() -> str:
    for name in ("FIREWORKS_API_KEY", "FIREWORKS_INFERENCE_API_KEY"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    raise SystemExit("Set FIREWORKS_API_KEY or FIREWORKS_INFERENCE_API_KEY")


def load_gemini_token() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_AI_API_KEY"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY")


def load_anthropic_token() -> str:
    token = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if token:
        return token
    raise SystemExit("Set ANTHROPIC_API_KEY")


def load_cases() -> list[dict]:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from load_panel import load_panel

    registry = json.loads((ROOT / "registry.json").read_text())
    cases = []
    for panel in registry["panels"]:
        _, panel_cases = load_panel(ROOT / panel["path"])
        cases.extend(panel_cases)
    return cases


def dataset_sha256(cases: list[dict]) -> str:
    """Hash canonical case content in registry order."""
    payload = json.dumps(
        cases,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def select_cases(cases: list[dict], smoke: bool, limit: int | None) -> list[dict]:
    if smoke:
        selected = []
        seen = set()
        for case in cases:
            if case["benchmark"] not in seen:
                selected.append(case)
                seen.add(case["benchmark"])
        cases = selected
    if limit is not None:
        cases = cases[:limit]
    return cases


def post_chat(
    token: str,
    model: str,
    messages: list[dict],
    max_completion_tokens: int,
    json_mode: bool,
    max_retries: int,
    reasoning_effort: str | None = None,
    timeout: int = 300,
    temperature: int | None = JUDGE_TEMPERATURE,
) -> tuple[str, int | None]:
    """Call Chat Completions at ``temperature`` (judge default 0), dropping it if rejected."""
    base = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if json_mode:
        base["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        base["reasoning_effort"] = reasoning_effort

    last_error = None
    for attempt in range(max_retries):
        body = dict(base)
        if temperature is not None:
            body["temperature"] = temperature
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"].get("content")
            if not content:
                raise RuntimeError("API returned an empty message")
            return content, temperature
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            last_error = RuntimeError(f"HTTP {error.code}: {detail}")
            if (
                error.code == 400
                and temperature is not None
                and "temperature" in detail.lower()
            ):
                temperature = None
                continue
            if error.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last_error
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error

        if attempt + 1 < max_retries:
            time.sleep(min(60, (2**attempt) + random.random()))

    raise RuntimeError(f"API request failed after {max_retries} attempts: {last_error}")


def is_gemini_model(model: str) -> bool:
    return resolve_answer_model(model).startswith("gemini")


def is_anthropic_model(model: str) -> bool:
    return resolve_answer_model(model).startswith("claude-")


def resolve_answer_model(model: str) -> str:
    """Map public model aliases to provider-specific model IDs."""
    aliases = {
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
        "glm-5.2-fast": "accounts/fireworks/routers/glm-5p2-fast",
        "glm-5p2-fast": "accounts/fireworks/routers/glm-5p2-fast",
        "kimi-k3": "accounts/fireworks/models/kimi-k3",
        "glm-5.3": "accounts/fireworks/models/glm-5p3",
        "glm-5p3": "accounts/fireworks/models/glm-5p3",
        "opus-5": "claude-opus-5",
        "claude-opus-5": "claude-opus-5",
    }
    return aliases.get(model, model)


def is_fireworks_model(model: str) -> bool:
    return resolve_answer_model(model).startswith("accounts/")


def post_fireworks(
    token: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    json_mode: bool,
    max_retries: int,
    reasoning_effort: str | None = None,
    timeout: int = 300,
    temperature: int | None = ANSWER_TEMPERATURE,
) -> tuple[str, int | None]:
    """Call Fireworks chat completions."""
    resolved = resolve_answer_model(model)
    requested = reasoning_effort or "none"
    base = {
        "model": resolved,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": requested,
    }
    if json_mode:
        base["response_format"] = {"type": "json_object"}

    fallback_low = requested == "none"
    last_error = None
    for attempt in range(max_retries):
        body = dict(base)
        if temperature is not None:
            body["temperature"] = temperature
        request = urllib.request.Request(
            FIREWORKS_API_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"].get("content")
            if not content:
                raise RuntimeError("Fireworks returned an empty message")
            return content, temperature
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            last_error = RuntimeError(f"HTTP {error.code}: {detail}")
            if (
                error.code == 400
                and fallback_low
                and body.get("reasoning_effort") == "none"
                and "reasoning_effort" in detail.lower()
            ):
                base["reasoning_effort"] = "low"
                continue
            if (
                error.code == 400
                and temperature is not None
                and "temperature" in detail.lower()
            ):
                temperature = None
                continue
            if error.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last_error
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error

        if attempt + 1 < max_retries:
            time.sleep(min(60, (2**attempt) + random.random()))

    raise RuntimeError(
        f"Fireworks request failed after {max_retries} attempts: {last_error}"
    )


def _anthropic_text(payload: dict) -> str:
    parts = []
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def post_anthropic(
    model: str,
    messages: list[dict],
    max_tokens: int,
    max_retries: int,
    reasoning_effort: str | None = None,
    timeout: int = 300,
) -> tuple[str, int | None]:
    """Call Anthropic Messages. Omit temperature (rejected by Claude 4.6+)."""
    token = load_anthropic_token()
    resolved = resolve_answer_model(model)
    system = ""
    non_system = []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        else:
            non_system.append(message)
    body = {
        "model": resolved,
        "max_tokens": max_tokens,
        "messages": non_system,
    }
    if system:
        body["system"] = system
    if reasoning_effort:
        body["output_config"] = {"effort": reasoning_effort}

    last_error = None
    for attempt in range(max_retries):
        request = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=json.dumps(body).encode(),
            headers={
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            content = _anthropic_text(payload)
            if not content:
                raise RuntimeError(
                    "Anthropic returned an empty text message "
                    f"stop={payload.get('stop_reason')}"
                )
            return content, None
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            last_error = RuntimeError(f"HTTP {error.code}: {detail}")
            if error.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last_error
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
        if attempt + 1 < max_retries:
            time.sleep(min(60, (2**attempt) + random.random()))

    raise RuntimeError(
        f"Anthropic request failed after {max_retries} attempts: {last_error}"
    )


def _gemini_text(payload: dict) -> str:
    candidate = (payload.get("candidates") or [{}])[0]
    parts = ((candidate.get("content") or {}).get("parts")) or []
    return "".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, dict) and part.get("text") and not part.get("thought")
    ).strip()


def _gemini_endpoint(model: str) -> tuple[str, dict]:
    """AI Studio API key, or Vertex service account via forge helpers."""
    resolved = resolve_answer_model(model)
    try:
        token = load_gemini_token()
    except SystemExit:
        token = ""
    if token:
        return GEMINI_API_URL.format(model=resolved) + f"?key={token}", {
            "Content-Type": "application/json"
        }
    from dox_llm_forge.llm_forge.clients.gemini_vertex import resolve_gemini_request

    return resolve_gemini_request(resolved)


def post_gemini(
    model: str,
    messages: list[dict],
    max_output_tokens: int,
    json_mode: bool,
    max_retries: int,
    reasoning_effort: str | None = None,
    timeout: int = 600,
    temperature: int | None = ANSWER_TEMPERATURE,
) -> tuple[str, int | None]:
    """Call Gemini generateContent (AI Studio or Vertex)."""
    contents = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        if message["role"] == "system":
            role = "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})

    generation_config: dict = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    if reasoning_effort:
        generation_config["thinkingConfig"] = {
            "thinkingLevel": reasoning_effort.upper(),
            "includeThoughts": False,
        }

    last_error = None
    for attempt in range(max_retries):
        url, headers = _gemini_endpoint(model)
        body_config = dict(generation_config)
        if temperature is None:
            body_config.pop("temperature", None)
        request = urllib.request.Request(
            url,
            data=json.dumps(
                {"contents": contents, "generationConfig": body_config}
            ).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = _gemini_text(json.load(response))
            if not content:
                raise RuntimeError("Gemini returned an empty message")
            return content, temperature
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            last_error = RuntimeError(f"HTTP {error.code}: {detail}")
            if (
                error.code == 400
                and temperature is not None
                and "temperature" in detail.lower()
            ):
                temperature = None
                continue
            if error.code not in (401, 408, 409, 429, 500, 502, 503, 504):
                raise last_error
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
        if attempt + 1 < max_retries:
            time.sleep(min(60, (2**attempt) + random.random()))

    raise RuntimeError(
        f"Gemini request failed after {max_retries} attempts: {last_error}"
    )


def render_judge_user(case: dict, answer: str) -> str:
    items = "\n".join(
        f"[{item['item_id']}] points={item['points']:+g} :: {item['text']}"
        for item in case["rubric_items"]
    )
    return (
        f"CLINICAL PROMPT:\n{case['prompt']}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"RUBRIC ITEMS:\n{items}"
    )


def parse_decisions(case: dict, content: str) -> dict[str, bool]:
    try:
        payload = json.loads(content)
        rows = payload["decisions"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid judge JSON: {error}") from error

    decisions = {}
    for row in rows:
        item_id = row.get("item_id")
        matched = row.get("matched")
        if not isinstance(item_id, str) or not isinstance(matched, bool):
            raise ValueError("each judge decision needs string item_id and bool matched")
        if item_id in decisions:
            raise ValueError(f"duplicate judge decision for {item_id}")
        decisions[item_id] = matched

    expected = {item["item_id"] for item in case["rubric_items"]}
    actual = set(decisions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"judge item mismatch; missing={missing}, extra={extra}")
    return decisions


def score_case(case: dict, decisions: dict[str, bool]) -> tuple[float, dict]:
    items = case["rubric_items"]
    if case["grading_method"] == "points_rubric":
        denominator = sum(max(float(item["points"]), 0.0) for item in items)
        if denominator <= 0:
            raise ValueError("points rubric has no positive-weight items")
        numerator = sum(
            float(item["points"])
            for item in items
            if decisions[item["item_id"]]
        )
        raw = numerator / denominator
        score = min(1.0, max(0.0, raw))
        return score, {
            "matched_signed_points": numerator,
            "available_positive_points": denominator,
            "raw_score": raw,
        }

    if case["grading_method"] == "f1_weighted_rubric":
        matched = [
            item["item_id"] for item in items if decisions[item["item_id"]]
        ]
        metrics = compute_f1_weighted(items, matched)
        return float(metrics["f1_weighted"] or 0.0), {
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "severe_rate": metrics["severe_rate"],
        }

    raise ValueError(f"unknown grading method: {case['grading_method']}")


def evaluate_case(
    token: str,
    case: dict,
    answer_model: str,
    judge_model: str,
    max_retries: int,
    reasoning_effort: str | None = None,
) -> dict:
    started = dt.datetime.now(dt.timezone.utc)
    answer_messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": case["prompt"]},
    ]
    answer_timeout = 600 if reasoning_effort else 300
    if is_gemini_model(answer_model):
        answer, answer_temperature = post_gemini(
            answer_model,
            answer_messages,
            max_output_tokens=ANSWER_MAX_TOKENS,
            json_mode=False,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
            timeout=max(answer_timeout, 600),
        )
    elif is_fireworks_model(answer_model):
        answer, answer_temperature = post_fireworks(
            load_fireworks_token(),
            answer_model,
            answer_messages,
            max_tokens=ANSWER_MAX_TOKENS,
            json_mode=False,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
            timeout=answer_timeout,
        )
    elif is_anthropic_model(answer_model):
        answer, answer_temperature = post_anthropic(
            answer_model,
            answer_messages,
            max_tokens=ANSWER_MAX_TOKENS,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
            timeout=answer_timeout,
        )
    else:
        answer, answer_temperature = post_chat(
            token,
            answer_model,
            answer_messages,
            max_completion_tokens=ANSWER_MAX_TOKENS,
            json_mode=False,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
            timeout=answer_timeout,
            temperature=ANSWER_TEMPERATURE,
        )

    judge_content, judge_temperature = post_chat(
        token,
        judge_model,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": render_judge_user(case, answer)},
        ],
        max_completion_tokens=JUDGE_MAX_TOKENS,
        json_mode=True,
        max_retries=max_retries,
    )
    decisions = parse_decisions(case, judge_content)
    score, components = score_case(case, decisions)
    finished = dt.datetime.now(dt.timezone.utc)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case["case_id"],
        "benchmark": case["benchmark"],
        "grading_method": case["grading_method"],
        "answer_model": answer_model,
        "judge_model": judge_model,
        "answer_reasoning_effort": reasoning_effort,
        "answer_temperature": answer_temperature,
        "judge_temperature": judge_temperature,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "answer": answer,
        "decisions": [
            {
                "item_id": item["item_id"],
                "points": item["points"],
                "matched": decisions[item["item_id"]],
            }
            for item in case["rubric_items"]
        ],
        "score": score,
        "score_components": components,
    }


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}") from error
    return rows


def successful_by_case(rows: list[dict]) -> dict[str, dict]:
    successful = {}
    for row in rows:
        if "score" in row and not row.get("error"):
            successful[row["case_id"]] = row
    return successful


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float], samples: int, rng: random.Random
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize(
    rows: list[dict],
    samples: int,
    seed: int,
    expected_case_ids: set[str] | None = None,
    dataset_digest: str | None = None,
) -> dict:
    successful = successful_by_case(rows)
    if expected_case_ids is not None:
        successful = {
            case_id: row
            for case_id, row in successful.items()
            if case_id in expected_case_ids
        }

    by_benchmark: dict[str, list[float]] = {}
    for row in successful.values():
        by_benchmark.setdefault(row["benchmark"], []).append(float(row["score"]))

    rng = random.Random(seed)
    benchmark_summary = {}
    for benchmark in sorted(by_benchmark):
        scores = by_benchmark[benchmark]
        low, high = bootstrap_mean_ci(scores, samples, rng)
        benchmark_summary[benchmark] = {
            "n": len(scores),
            "mean": statistics.fmean(scores),
            "ci95": [low, high],
        }

    overall = math.nan
    overall_ci = [math.nan, math.nan]
    if by_benchmark:
        overall = statistics.fmean(
            statistics.fmean(scores) for scores in by_benchmark.values()
        )
        bootstrapped = []
        names = sorted(by_benchmark)
        for _ in range(samples):
            means = []
            for name in names:
                scores = by_benchmark[name]
                n = len(scores)
                means.append(
                    sum(scores[rng.randrange(n)] for _ in range(n)) / n
                )
            bootstrapped.append(statistics.fmean(means))
        overall_ci = [
            percentile(bootstrapped, 0.025),
            percentile(bootstrapped, 0.975),
        ]

    models = {
        (row.get("answer_model"), row.get("judge_model"))
        for row in successful.values()
    }
    temperatures = {
        (row.get("answer_temperature"), row.get("judge_temperature"))
        for row in successful.values()
    }
    relevant_rows = [
        row
        for row in rows
        if expected_case_ids is None or row.get("case_id") in expected_case_ids
    ]
    latest_by_case = {row.get("case_id"): row for row in relevant_rows}
    started_at = [row["started_at"] for row in relevant_rows if row.get("started_at")]
    finished_at = [
        row["finished_at"] for row in relevant_rows if row.get("finished_at")
    ]
    temperature_pair = next(iter(temperatures)) if len(temperatures) == 1 else (None, None)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_sha256": dataset_digest,
        "answer_model": next(iter(models))[0] if len(models) == 1 else None,
        "judge_model": next(iter(models))[1] if len(models) == 1 else None,
        "answer_temperature": temperature_pair[0],
        "judge_temperature": temperature_pair[1],
        "judge_passes_per_answer": 1,
        "started_at": min(started_at) if started_at else None,
        "finished_at": max(finished_at) if finished_at else None,
        "completed_cases": len(successful),
        "errors": sum(bool(row.get("error")) for row in latest_by_case.values()),
        "overall_macro_mean": overall,
        "overall_ci95": overall_ci,
        "benchmarks": benchmark_summary,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def validate_existing_configuration(
    rows: list[dict], answer_model: str, judge_model: str
) -> None:
    for row in rows:
        if row.get("answer_model") not in (None, answer_model):
            raise SystemExit(
                f"output contains answer_model={row.get('answer_model')!r}; "
                f"requested {answer_model!r}"
            )
        if row.get("judge_model") not in (None, judge_model):
            raise SystemExit(
                f"output contains judge_model={row.get('judge_model')!r}; "
                f"requested {judge_model!r}"
            )
        if row.get("protocol_version") not in (None, PROTOCOL_VERSION):
            raise SystemExit("output contains results from another protocol version")


def main() -> int:
    args = parse_args()
    all_cases = load_cases()
    dataset_digest = dataset_sha256(all_cases)
    selected = select_cases(all_cases, args.smoke, args.limit)
    expected_ids = {case["case_id"] for case in selected}
    rows = load_results(args.output)
    validate_existing_configuration(rows, args.answer_model, args.judge_model)

    if args.summarize_only:
        summary = summarize(
            rows,
            args.bootstrap_samples,
            args.bootstrap_seed,
            expected_ids,
            dataset_digest,
        )
        print(json.dumps(summary, indent=2))
        return 0

    token = load_token()
    completed = successful_by_case(rows)
    pending = [case for case in selected if case["case_id"] not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    answer_effort = resolve_reasoning_effort(args.answer_model, args.reasoning_effort)

    print(
        f"selected={len(selected)} completed={len(selected) - len(pending)} "
        f"pending={len(pending)} workers={args.workers} "
        f"reasoning_effort={answer_effort}"
    )

    errors = 0
    with args.output.open("a", buffering=1) as output:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers)
        ) as executor:
            futures = {
                executor.submit(
                    evaluate_case,
                    token,
                    case,
                    args.answer_model,
                    args.judge_model,
                    args.max_retries,
                    answer_effort,
                ): case
                for case in pending
            }
            finished = 0
            for future in concurrent.futures.as_completed(futures):
                case = futures[future]
                try:
                    row = future.result()
                    status = f"score={row['score']:.4f}"
                except Exception as error:  # preserve failure for audit and resume
                    errors += 1
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "case_id": case["case_id"],
                        "benchmark": case["benchmark"],
                        "grading_method": case["grading_method"],
                        "answer_model": args.answer_model,
                        "judge_model": args.judge_model,
                        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "error": f"{type(error).__name__}: {error}"[:2000],
                    }
                    status = "ERROR"
                with write_lock:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                finished += 1
                print(
                    f"[{finished}/{len(pending)}] {case['benchmark']}/"
                    f"{case['case_id'][:8]} {status}",
                    flush=True,
                )

    rows = load_results(args.output)
    summary = summarize(
        rows,
        args.bootstrap_samples,
        args.bootstrap_seed,
        expected_ids,
        dataset_digest,
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
