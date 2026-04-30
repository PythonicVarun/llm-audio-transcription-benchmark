#!/usr/bin/env python3
"""
Manual transcript benchmark runner.

This script reads model transcripts from files and evaluates them using the same
metrics and evaluator prompts as benchmark.py.
"""

import argparse
import base64
import json
import logging
import pathlib
from typing import Optional

from tqdm import tqdm

from benchmark import (
    EVALUATOR_MODEL,
    MIME_MAP,
    OUTPUT_PATH,
    build_report,
    collect_model_keys,
    compute_aggregates,
    compute_metrics_for_output,
    evaluate_one,
    generate_summary,
    load_manifest,
    recompute_result_metrics,
    sync_summary_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MANUAL_TRANSCRIPTION_MODELS: dict[str, dict[str, str]] = {
    "openai-whisper-20250625-large": {
        "display": "OpenAI Whisper (Large)",
        "suffix": ".whisper.txt",
    },
    "gemini-3.1-pro-preview": {
        "display": "Gemini 3.1 Pro Preview",
        "suffix": ".gem_3_1_pro.txt",
    },
    "gemini-3-flash-preview": {
        "display": "Gemini 3 Flash Preview",
        "suffix": ".gem_3_flash_pre.txt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark from manual transcript files only. "
            "Expected transcript suffixes: .whisper.txt, .gem_3_1_pro.txt, .gem_3_flash_pre.txt"
        )
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=OUTPUT_PATH,
        help="Output report path (default: benchmark_report.json).",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "Optional comma-separated manual model keys to run. "
            "Available: "
            + ", ".join(MANUAL_TRANSCRIPTION_MODELS)
        ),
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        default=False,
        help=(
            "Skip per-sample LLM evaluation and global summary generation. "
            "WER/CER/MER metrics are still computed and written."
        ),
    )
    return parser.parse_args()


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_manual_transcript(audio_path: pathlib.Path, suffix: str) -> tuple[str, pathlib.Path]:
    transcript_path = audio_path.parent / f"{audio_path.stem}{suffix}"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Manual transcript not found: {transcript_path}")
    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript:
        raise ValueError(f"Manual transcript file is empty: {transcript_path}")
    return transcript, transcript_path


def run_manual_benchmark(
    output_path: pathlib.Path,
    model_filter: Optional[list[str]] = None,
    skip_evaluation: bool = False,
) -> None:
    selected_models = dict(MANUAL_TRANSCRIPTION_MODELS)
    if model_filter:
        missing = [key for key in model_filter if key not in selected_models]
        if missing:
            raise ValueError(
                "Unknown model key(s) in --models: "
                + ", ".join(missing)
                + ". Available: "
                + ", ".join(selected_models)
            )
        selected_models = {key: selected_models[key] for key in model_filter}

    manifest = load_manifest()
    logger.info("Loaded %s audio samples from manifest.json", len(manifest))
    if model_filter:
        logger.info("Model filter enabled: %s", ", ".join(selected_models))

    missing_audio = [e["file"] for e in manifest if not pathlib.Path(e["file"]).exists()]
    if missing_audio:
        logger.warning("Missing audio files (will be skipped):\n  %s", "\n  ".join(missing_audio))

    all_results: list[dict] = []

    for entry in tqdm(manifest, desc="Audio samples", unit="file"):
        audio_path = pathlib.Path(entry["file"])
        if not audio_path.exists():
            logger.warning("Skipping %s — file not found: %s", entry["id"], audio_path)
            continue

        audio_id = entry["id"]
        variation = entry["variation"]
        reference = entry.get("reference")
        reference_path = entry.get("reference_path")
        if reference_path:
            ref_path = pathlib.Path(reference_path)
            if ref_path.exists():
                reference = ref_path.read_text(encoding="utf-8").strip()
            else:
                logger.warning(
                    "Reference path not found for %s: %s (using inline reference if available)",
                    audio_id,
                    ref_path,
                )

        language = entry.get("language", "en")

        logger.info("\n%s", "─" * 65)
        logger.info("▶ [%s]  %s  (%s)", audio_id, entry["dataset"], entry["variation_label"])

        model_outputs: dict[str, dict] = {}
        for model_key, model_cfg in selected_models.items():
            logger.info("  ▷ Loading manual transcript for %s ...", model_cfg["display"])
            try:
                transcript, transcript_path = load_manual_transcript(
                    audio_path=audio_path,
                    suffix=model_cfg["suffix"],
                )
                error = None
                source = f"manual_file:{transcript_path}"
            except Exception as exc:
                transcript = ""
                error = str(exc)
                source = "error"
                logger.error("  ✗ [%s] %s → %s", model_key, audio_path.name, exc)

            wer, cer, mer = compute_metrics_for_output(reference, transcript, language)
            model_outputs[model_key] = {
                "transcript": transcript,
                "wer": wer,
                "cer": cer,
                "mer": mer,
                "latency_ms": 0.0,
                "error": error,
                "transcription_source": source,
            }

            wer_str = f"WER={wer:.1%}" if wer is not None else "WER=N/A"
            logger.info(
                "    %s  CER=%.1f%%  latency=0ms" % (wer_str, cer * 100)
                if cer is not None
                else f"    {wer_str}  latency=0ms"
            )

        evaluations: dict[str, dict] = {}
        if skip_evaluation:
            logger.info("  ▷ Skipping evaluation (--skip-evaluation)")
        else:
            logger.info("  ▷ Evaluating all outputs with %s ...", EVALUATOR_MODEL)
            for model_key in selected_models:
                out = model_outputs[model_key]
                evaluations[model_key] = evaluate_one(
                    audio_id=audio_id,
                    variation=variation,
                    language=language,
                    model_key=model_key,
                    reference=reference,
                    hypothesis=out["transcript"],
                    wer=out["wer"],
                    cer=out["cer"],
                    mer=out["mer"],
                )
                score = evaluations[model_key].get("overall_score", "?")
                logger.info("    %-30s → score=%s/10", selected_models[model_key]["display"], score)

        result = {
            "audio_id": audio_id,
            "variation": variation,
            "variation_label": entry.get("variation_label", variation),
            "dataset": entry["dataset"],
            "language": language,
            "audio_file": str(audio_path),
            "audio_mime_type": MIME_MAP.get(audio_path.suffix.lower(), "audio/wav"),
            "audio_base64": base64.b64encode(audio_path.read_bytes()).decode(),
            "reference_transcript": reference,
            "model_outputs": model_outputs,
            "evaluations": evaluations,
        }
        if "accent_origin" in entry:
            result["accent_origin"] = entry["accent_origin"]

        recompute_result_metrics(result)
        all_results.append(result)

    transcription_models = {mk: cfg["display"] for mk, cfg in selected_models.items()}
    transcription_model_details = {
        mk: {
            "provider": "manual_file",
            "display": cfg["display"],
            "manual_suffix": cfg["suffix"],
        }
        for mk, cfg in selected_models.items()
    }

    model_keys = collect_model_keys(all_results, list(transcription_models))
    for model_key in model_keys:
        transcription_models.setdefault(model_key, model_key)

    aggregates = compute_aggregates(all_results, model_keys=model_keys)
    if skip_evaluation:
        logger.info("▶ Skipping summary generation (--skip-evaluation)")
        summary = {}
    else:
        logger.info("▶ Generating benchmark summary with %s ...", EVALUATOR_MODEL)
        summary = generate_summary(all_results, model_keys=model_keys)
        sync_summary_metrics(summary, aggregates)

    report = build_report(
        all_results=all_results,
        summary=summary,
        aggregates=aggregates,
        transcription_models=transcription_models,
        transcription_model_details=transcription_model_details,
        append_report=False,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)

    logger.info("\n%s", "═" * 65)
    logger.info("✅  Manual benchmark complete → %s", output_path.resolve())
    logger.info("    Samples processed : %s", len(all_results))
    logger.info("%s\n", "═" * 65)


if __name__ == "__main__":
    args = parse_args()
    run_manual_benchmark(
        output_path=args.output_path,
        model_filter=_split_csv(args.models),
        skip_evaluation=args.skip_evaluation,
    )
