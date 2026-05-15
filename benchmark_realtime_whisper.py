#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          gpt-realtime-whisper   ASR Benchmark Addon              ║
║                                                                  ║
║  Transcribes every audio file in manifest.json using the         ║
║  OpenAI Realtime Transcription WebSocket API, evaluates each     ║
║  result with the same EVALUATOR_MODEL used in benchmark.py,      ║
║  and APPENDS the new model's outputs to benchmark_report.json    ║
║  without touching any previously stored model results.           ║
╚══════════════════════════════════════════════════════════════════╝

Realtime Transcription protocol (from OpenAI docs)
───────────────────────────────────────────────────
  1. Connect    wss://api.openai.com/v1/realtime?intent=transcription
                headers: Authorization: Bearer <key>

  2. Receive    { "type": "session.created" }

  3. Send       { "type": "session.update",
                  "session": {
                    "type": "transcription",
                    "audio": {
                      "input": {
                        "format":        { "type": "audio/pcm", "rate": 24000 },
                        "transcription": { "model": "gpt-realtime-whisper",
                                           "language": "en" },
                        "turn_detection": null          ← manual-commit mode
                      }
                    }
                  }
                }

  4. Receive    { "type": "session.updated" }

  5. Send       { "type": "input_audio_buffer.append",
                  "audio": "<base64 PCM-16 mono 24 kHz chunk>" }
                  … (repeat for every chunk) …

  6. Send       { "type": "input_audio_buffer.commit" }

  7. Receive    { "type": "conversation.item.input_audio_transcription.delta",
                  "delta": "Hello," }
                  … (stream of partial text) …

  8. Receive    { "type": "conversation.item.input_audio_transcription.completed",
                  "transcript": "Hello, how are you?" }

Audio requirements
──────────────────
  Raw PCM-16, mono, 24 000 Hz.
  Any input format is converted automatically:
    • soundfile + scipy.signal - handles WAV / FLAC / OGG / AIFF
    • ffmpeg subprocess fallback - handles MP3 / AAC / M4A / anything else

Dependencies (beyond benchmark.py requirements)
────────────────────────────────────────────────
  pip install websockets soundfile scipy numpy

Usage
─────
  python benchmark_whisper.py                     # append to existing report
  python benchmark_whisper.py --skip-evaluation   # skip LLM eval step
  python benchmark_whisper.py --overwrite         # replace instead of append
  python benchmark_whisper.py --output <path>     # custom report path
  python benchmark_whisper.py --ws-url  <url>     # override WebSocket endpoint
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import traceback
from math import gcd
from typing import Optional

import websockets

_HERE = pathlib.Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from benchmark import (
    EVALUATOR_MODEL,
    MANIFEST_PATH,
    MIME_MAP,
    OPENAI_API_KEY,
    OUTPUT_PATH,
    TRANSCRIPTION_MODELS,
    build_report,
    collect_model_keys,
    compute_aggregates,
    compute_metrics_for_output,
    evaluate_one,
    generate_summary,
    load_existing_report,
    load_manifest,
    merge_results,
    merge_transcription_model_metadata,
    recompute_result_metrics,
    sync_summary_metrics,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model being benchmarked
MODEL_KEY = "gpt-realtime-whisper"
MODEL_ID = "gpt-realtime-whisper"
MODEL_DISPLAY = "GPT Realtime Whisper"

TRANSCRIPTION_MODELS[MODEL_KEY] = {
    "provider": "openai_realtime",
    "model_id": MODEL_ID,
    "display": MODEL_DISPLAY,
}

DEFAULT_WS_URL: str = "wss://api.openai.com/v1/realtime?intent=transcription"

# Chunk size: how many milliseconds of PCM-16 to send per WebSocket frame.
# 100 ms -> 4 800 bytes at 24 kHz mono 16-bit.
CHUNK_MS: int = int(os.getenv("REALTIME_CHUNK_MS", "100"))

# Per-file receive timeout = base + multiplier × audio_duration_seconds
TIMEOUT_BASE_S: float = 180.0
TIMEOUT_MULTIPLIER: float = 3.0

SECONDS_PER_COMMIT = 120

BYTES_PER_SECOND = 24_000 * 2  # 24kHz * int16
COMMIT_BYTES = BYTES_PER_SECOND * SECONDS_PER_COMMIT


# Audio conversion helpers
# Goal: raw PCM-16, mono, 24 000 Hz bytes
def stream_pcm16_24k(path: pathlib.Path, frame_bytes: int):
    """
    Stream PCM16 mono 24k audio directly from ffmpeg stdout.
    Prevents loading huge audio files into RAM.
    """
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "24000",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    try:
        while True:
            chunk = process.stdout.read(frame_bytes)

            if not chunk:
                break

            yield chunk
    finally:
        if process.stdout:
            process.stdout.close()

        process.kill()
        process.wait()


# Realtime Transcription WebSocket session (async)
async def _recv_until(ws, target_types: set, timeout: float = 60.0) -> dict:
    """
    Consume incoming WebSocket messages until one whose 'type' is in
    *target_types* arrives.  Raises TimeoutError or RuntimeError on error events.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for any of {target_types}")
        raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
        event = json.loads(raw)
        etype = event.get("type", "")
        if etype in target_types:
            return event
        if etype == "error":
            raise RuntimeError(
                f"Realtime API returned an error event: {json.dumps(event)}"
            )
        logger.debug("    ← [ignored] %s", etype)


async def _realtime_transcribe(
    audio_path: pathlib.Path,
    language: str,
    api_key: str,
    ws_url: str,
    chunk_ms: int = CHUNK_MS,
) -> str:
    """
    Core async routine.  Opens a Realtime transcription WebSocket session,
    streams the entire audio file, and returns the final transcript string.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    transcript_deltas: list[str] = []

    async with websockets.connect(
        ws_url,
        additional_headers=headers,
        open_timeout=180,
        close_timeout=180,
    ) as ws:

        # Step 1: wait for the server to acknowledge the connection
        await _recv_until(ws, {"session.created"}, timeout=180.0)
        logger.debug("    ← session.created")

        # Step 2: configure a transcription session
        #
        # Shape taken directly from the OpenAI Realtime Transcription docs:
        #
        #  session.type                               = "transcription"
        #  session.audio.input.format.type            = "audio/pcm"
        #  session.audio.input.format.rate            = 24000
        #  session.audio.input.transcription.model    = "gpt-realtime-whisper"
        #  session.audio.input.transcription.language = <iso code>
        #  session.audio.input.turn_detection         = null  (manual commit)
        #
        session_cfg: dict = {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24_000,
                    },
                    "transcription": {
                        "model": MODEL_ID,
                        "language": language,
                    },
                }
            },
        }

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": session_cfg,
                }
            )
        )

        # Step 3: wait for the server to confirm the session config
        await _recv_until(
            ws,
            {"session.updated", "transcription_session.updated"},
            timeout=180.0,
        )
        logger.debug("    ← session.updated")

        # Step 4: stream audio in fixed-size PCM-16 chunks

        frame_bytes = int(24_000 * 2 * chunk_ms / 1_000)
        sent_since_commit = 0
        total_audio_bytes = 0

        for chunk in stream_pcm16_24k(audio_path, frame_bytes):
            total_audio_bytes += len(chunk)

            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )

            sent_since_commit += len(chunk)

            # Commit every 120 seconds
            if sent_since_commit >= COMMIT_BYTES:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                logger.info(
                    "    -> Committed %.1f sec", sent_since_commit / (24_000 * 2)
                )

                sent_since_commit = 0
                partial_deltas = []

                # Wait for transcript completion
                while True:
                    raw = await ws.recv()
                    event = json.loads(raw)

                    etype = event.get("type", "")

                    if etype == "conversation.item.input_audio_transcription.delta":
                        delta = event.get("delta", "")
                        if delta:
                            partial_deltas.append(delta)
                    elif (
                        etype == "conversation.item.input_audio_transcription.completed"
                    ):
                        delta = event.get("transcript", "").strip()
                        if delta:
                            transcript_deltas.append(delta)
                        else:
                            transcript_deltas.extend("".join(partial_deltas))

                        break
                    elif etype == "error":
                        raise RuntimeError(f"Realtime API error: {json.dumps(event)}")

        # Step 5: commit the buffer to trigger transcription
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        logger.debug("    -> Final input_audio_buffer.commit")

        # Step 6: collect transcript events
        partial_deltas = []
        while True:
            try:
                raw = await ws.recv()
            except asyncio.TimeoutError:
                # No message for 5 s - loop back and check the overall deadline
                continue

            event = json.loads(raw)
            etype = event.get("type", "")
            logger.debug("    ← %s", etype)

            if etype == "conversation.item.input_audio_transcription.delta":
                delta = event.get("delta", "")
                if delta:
                    partial_deltas.append(delta)

            elif etype == "conversation.item.input_audio_transcription.completed":
                # The 'transcript' field on the completed event is authoritative:
                # it is the server's final, corrected version and may differ from
                # the concatenation of deltas (earlier partials can be revised).
                final = (event.get("transcript") or "").strip()
                if final:
                    transcript_deltas.append(final)
                else:
                    transcript_deltas.extend("".join(partial_deltas))

                break

            elif etype == "error":
                raise RuntimeError(f"Realtime API error: {json.dumps(event)}")

    return "\n".join(transcript_deltas).strip()


# Synchronous transcription wrapper


def transcribe(
    audio_path: pathlib.Path,
    language: str,
    ws_url: str,
) -> tuple[str, float]:
    """
    Transcribe *audio_path* and return (transcript, latency_ms).
    The prompt is resolved from the variation tag using the same helper as
    benchmark.py so prompting stays consistent across models.
    """
    t0 = time.monotonic()
    text = asyncio.run(
        _realtime_transcribe(
            audio_path=audio_path,
            language=language,
            api_key=OPENAI_API_KEY,
            ws_url=ws_url,
        )
    )
    latency_ms = (time.monotonic() - t0) * 1_000
    return text, latency_ms


def safe_transcribe(
    audio_path: pathlib.Path,
    language: str,
    ws_url: str,
) -> tuple[str, float, Optional[str]]:
    """
    Wrapper that catches all exceptions and returns
    (transcript, latency_ms, error_string | None).
    """
    try:
        text, latency_ms = transcribe(audio_path, language, ws_url)
        return text, latency_ms, None
    except Exception as exc:
        traceback.print_exc()
        logger.error("  ✗ [%s] %s — %s", MODEL_KEY, audio_path.name, exc)
        return "", 0.0, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark gpt-realtime-whisper on every file in manifest.json "
            "via the OpenAI Realtime Transcription WebSocket API and append "
            "the results to benchmark_report.json."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help=(
            "Replace benchmark_report.json instead of merging into it. "
            "Default is APPEND (all existing model results are preserved)."
        ),
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        default=False,
        help=(
            "Skip per-sample LLM evaluation and global summary generation. "
            "Transcripts and WER/CER/MER metrics are still written."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        metavar="PATH",
        help=f"Path to the benchmark report JSON (default: {OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_URL,
        metavar="URL",
        help=(
            "WebSocket URL for the Realtime API.  "
            f"Defaults to {DEFAULT_WS_URL}.  "
            "Can also be set via the OPENAI_REALTIME_WS_URL env var."
        ),
    )
    return parser.parse_args()


def run(
    append_report: bool = True,
    skip_evaluation: bool = False,
    output_path: pathlib.Path = OUTPUT_PATH,
    ws_url: str = DEFAULT_WS_URL,
) -> None:
    manifest = load_manifest()
    logger.info("Loaded %d audio samples from %s", len(manifest), MANIFEST_PATH)
    logger.info("WebSocket URL  : %s", ws_url)
    logger.info("Evaluator model: %s", EVALUATOR_MODEL)

    # Load existing report (if appending)
    existing_report: Optional[dict] = None
    if append_report:
        existing_report = load_existing_report(output_path)
        if existing_report:
            logger.info("Append mode - will merge into: %s", output_path.resolve())
        else:
            logger.info("Append mode - no existing report found; creating a new one.")

    # Warn about missing audio files up-front
    missing = [e["file"] for e in manifest if not pathlib.Path(e["file"]).exists()]
    if missing:
        logger.warning(
            "%d audio file(s) not found (will be skipped):\n  %s",
            len(missing),
            "\n  ".join(missing),
        )

    all_results: list[dict] = []

    for entry in manifest:
        audio_path = pathlib.Path(entry["file"])
        if not audio_path.exists():
            logger.warning("  ⚠ Skipping %s - not found", entry["id"])
            continue

        audio_id = entry["id"]
        variation = entry["variation"]
        language = entry.get("language", "en")

        # Resolve reference transcript
        reference: Optional[str] = entry.get("reference")
        ref_path_str = entry.get("reference_path")
        if ref_path_str:
            ref_path = pathlib.Path(ref_path_str)
            if ref_path.exists():
                reference = ref_path.read_text(encoding="utf-8").strip()
            else:
                logger.warning(
                    "  ⚠ Reference path not found for %s: %s",
                    audio_id,
                    ref_path,
                )

        logger.info("\n%s", "─" * 66)
        logger.info(
            "▶ [%s]  dataset=%s  variation=%s  lang=%s",
            audio_id,
            entry["dataset"],
            entry["variation_label"],
            language,
        )

        path_parts = list(audio_path.with_suffix(".realtime_whisper.txt").parts)
        path_parts[path_parts.index("dataset")] = "transcripts"
        transcript_path = pathlib.Path(*path_parts)

        # 1. Transcribe
        if transcript_path.exists():
            logger.info(
                "  ▷ Transcript already exists at %s, skipping transcription.",
                transcript_path.resolve(),
            )
            transcript = transcript_path.read_text(encoding="utf-8").strip()
            latency_ms = 0.0
            error = None
        else:
            logger.info("  ▷ Transcribing with %s …", MODEL_DISPLAY)
            transcript, latency_ms, error = safe_transcribe(
                audio_path, language, ws_url
            )

            # Save the trascript as .realtime_whisper.txt in transcripts/
            transcript_path.write_text(transcript, encoding="utf-8")
            logger.info("    Transcript saved to: %s", transcript_path.resolve())

        wer, cer, mer = compute_metrics_for_output(
            reference or "", transcript, language
        )

        model_outputs: dict = {
            MODEL_KEY: {
                "transcript": transcript,
                "wer": wer,
                "cer": cer,
                "mer": mer,
                "latency_ms": round(latency_ms, 1),
                "error": error,
                "transcription_source": "realtime_ws",
            }
        }

        logger.info(
            "    WER=%-6s  CER=%-6s  latency=%.0f ms",
            f"{wer:.1%}" if wer is not None else "N/A",
            f"{cer:.1%}" if cer is not None else "N/A",
            latency_ms,
        )

        # 2. LLM evaluation
        # Uses the exact same evaluate_one() call as benchmark.py — same
        # EVALUATOR_MODEL, same prompts, same scoring rubric.
        evaluations: dict = {}
        if skip_evaluation:
            logger.info("  ▷ Evaluation skipped (--skip-evaluation)")
        else:
            logger.info("  ▷ Evaluating with %s …", EVALUATOR_MODEL)
            out = model_outputs[MODEL_KEY]
            evaluations[MODEL_KEY] = evaluate_one(
                audio_id=audio_id,
                variation=variation,
                language=language,
                model_key=MODEL_KEY,
                reference=reference or "",
                hypothesis=out["transcript"],
                wer=out["wer"],
                cer=out["cer"],
                mer=out["mer"],
            )
            score = evaluations[MODEL_KEY].get("overall_score", "?")
            logger.info("    %s -> score = %s / 10", MODEL_DISPLAY, score)

        # Build result record (same schema as benchmark.py)
        result: dict = {
            "audio_id": audio_id,
            "variation": variation,
            "variation_label": entry.get("variation_label", variation),
            "dataset": entry["dataset"],
            "language": language,
            "audio_file": str(audio_path),
            "audio_mime_type": MIME_MAP.get(audio_path.suffix.lower(), "audio/wav"),
            "audio_base64": None,
            "reference_transcript": reference,
            "model_outputs": model_outputs,
            "evaluations": evaluations,
        }
        if "accent_origin" in entry:
            result["accent_origin"] = entry["accent_origin"]

        recompute_result_metrics(result)
        all_results.append(result)

    # 3. Merge with existing report
    merge_stats: Optional[dict] = None
    previous_report_timestamp: Optional[str] = None

    if append_report and existing_report:
        previous_report_timestamp = existing_report.get("benchmark_metadata", {}).get(
            "timestamp"
        )
        all_results, merge_stats = merge_results(
            existing_report.get("results", []),
            all_results,
        )
        # Recompute derived metrics for every row so they stay in sync
        for r in all_results:
            recompute_result_metrics(r)
        logger.info(
            "Merged - rows: %d added, %d updated | "
            "model outputs: %d added, %d updated",
            merge_stats["audio_rows_added"],
            merge_stats["audio_rows_updated"],
            merge_stats["model_outputs_added"],
            merge_stats["model_outputs_updated"],
        )

    # 4. Aggregates + global summary
    transcription_models, transcription_model_details = (
        merge_transcription_model_metadata(existing_report if append_report else None)
    )
    model_keys = collect_model_keys(all_results, list(transcription_models))
    for mk in model_keys:
        transcription_models.setdefault(mk, mk)

    logger.info("\n%s", "─" * 66)
    aggregates = compute_aggregates(all_results, model_keys=model_keys)

    if skip_evaluation:
        logger.info("▶ Summary generation skipped (--skip-evaluation)")
        summary: dict = {}
    else:
        logger.info("▶ Generating summary with %s …", EVALUATOR_MODEL)
        summary = generate_summary(all_results, model_keys=model_keys)
        sync_summary_metrics(summary, aggregates)

    # 5. Write report
    report = build_report(
        all_results=all_results,
        summary=summary,
        aggregates=aggregates,
        transcription_models=transcription_models,
        transcription_model_details=transcription_model_details,
        append_report=append_report,
        merge_stats=merge_stats,
        previous_report_timestamp=previous_report_timestamp,
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # Console summary table
    col = {"model": 35, "wer": 9, "score": 10, "lat": 10}
    sep = f"    {'─'*col['model']}  {'─'*col['wer']}  {'─'*col['score']}  {'─'*col['lat']}"
    logger.info("\n%s", "═" * 66)
    logger.info("✅  Written -> %s", output_path.resolve())
    logger.info("    Total samples in report: %d", len(all_results))
    logger.info(
        f"    {'Model':<{col['model']}}  {'Avg WER':>{col['wer']}}  "
        f"{'Avg Score':>{col['score']}}  {'Avg Lat':>{col['lat']}}"
    )
    logger.info(sep)
    for mk, agg in aggregates.items():
        display = str(transcription_models.get(mk, mk))
        wer_s = f"{agg['avg_wer']:.1%}" if agg["avg_wer"] is not None else "N/A"
        score_s = (
            f"{agg['avg_eval_score']:.1f}/10"
            if agg["avg_eval_score"] is not None
            else "N/A"
        )
        lat_s = (
            f"{agg['avg_latency_ms']:.0f} ms"
            if agg["avg_latency_ms"] is not None
            else "N/A"
        )
        logger.info(
            f"    {display:<{col['model']}}  {wer_s:>{col['wer']}}  "
            f"{score_s:>{col['score']}}  {lat_s:>{col['lat']}}"
        )
    logger.info("%s\n", "═" * 66)


if __name__ == "__main__":
    args = parse_args()
    run(
        append_report=not args.overwrite,
        skip_evaluation=args.skip_evaluation,
        output_path=pathlib.Path(args.output),
        ws_url=args.ws_url,
    )
