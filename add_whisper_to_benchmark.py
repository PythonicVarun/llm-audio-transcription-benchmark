#!/usr/bin/env python3
"""Add Whisper transcripts into an existing benchmark_report.json.

Expected transcript file convention (from whisper.sh):
  <audio_stem>.whisper.txt
stored alongside each audio file.

Example:
  python add_whisper_to_benchmark.py --report benchmark_report.json
  python add_whisper_to_benchmark.py --report benchmark_report.json --enable-evaluation
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import jiwer
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("CODEX_STRAIVE_OPENAI_TOKEN", "")
OPENAI_BASE_URL = (
    os.getenv("OPENAI_BASE_URL", "https://llmfoundry.straivedemo.com/openai/v1").strip()
    or None
)
DEFAULT_EVALUATOR_MODEL = "gpt-5.5"

openai_client: Optional[OpenAI] = None

_EVAL_SYSTEM = """\
You are an expert automatic speech recognition (ASR) evaluator.

Given:
- A reference transcript (ground truth)
- A hypothesis transcript (model output)
- Metadata about the audio variation and model

Your task is to produce a structured quality evaluation as JSON.

Be strict. The reference transcript is authoritative. Penalize every
meaning-changing substitution, deletion, insertion, hallucination, language
mix-up, speaker attribution error, and word-order error. Do not give credit for
fluent or plausible text if it does not match the reference.

Scoring rules:
- Start from 10 and reduce scores based on concrete transcript errors.
- Perfect or near-perfect transcripts only: 9-10.
- Minor wording, punctuation, casing, or formatting differences only: 8.
- Noticeable word substitutions/deletions/insertions that preserve most meaning: 6-7.
- Repeated errors, missed phrases, wrong names/numbers, or partial hallucination: 4-5.
- Large omissions, frequent hallucinations, wrong language, or mostly wrong transcript: 1-3.
- Empty output is always 0 and should be handled as complete failure.

Hard caps:
- If WER is provided and is greater than 20%, overall_score must be at most 7.
- If WER is provided and is greater than 40%, overall_score must be at most 5.
- If WER is provided and is greater than 60%, overall_score must be at most 3.
- If the hypothesis misses more than one third of the reference, completeness_score must be at most 5.
- If the hypothesis includes substantial hallucinated content, accuracy_score and overall_score must be at most 5.
- If the language is wrong or mixed in a way that changes the transcript, overall_score must be at most 4.
- For multiple_speakers samples, missing speaker labels, inconsistent speaker labels, or wrong speaker turns must be penalized under speaker_overlap_confusion; severe speaker-format failure caps overall_score at 6.
- Do not use error_categories ["none"] unless the transcript is essentially error-free.

Respond ONLY with a valid JSON object using EXACTLY this schema — no markdown, no extra keys:
{
  "accuracy_score":       <integer 1-10>,
  "fluency_score":        <integer 1-10>,
  "completeness_score":   <integer 1-10>,
  "overall_score":        <integer 1-10>,
  "failure_summary":      "<one to three sentences describing what went wrong>",
  "error_categories":     ["<category>"],
  "improvement_suggestions": "<concrete, actionable suggestions>",
  "notable_errors":       ["<verbatim error example>"]
}

Available error_categories (pick all that apply):
  substitution_errors, deletion_errors, insertion_errors,
  accent_confusion, language_confusion, speaker_overlap_confusion,
  noise_interference, hallucination, punctuation_errors,
  proper_noun_errors, foreign_word_confusion, none
"""

_EMPTY_EVAL = {
    "accuracy_score": 0,
    "fluency_score": 0,
    "completeness_score": 0,
    "overall_score": 0,
    "failure_summary": "Model returned empty output — complete transcription failure.",
    "error_categories": ["deletion_errors"],
    "improvement_suggestions": (
        "Model produced no output for this audio. Check audio format compatibility, "
        "file size limits, and whether the model supports this audio codec/language."
    ),
    "notable_errors": ["Entire transcription missing"],
}

_WER_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ]
)


def compute_wer(reference: str, hypothesis: str) -> Optional[float]:
    if not reference.strip() or not hypothesis.strip():
        return None
    try:
        return round(
            jiwer.wer(
                reference,
                hypothesis,
                reference_transform=_WER_TRANSFORM,
                hypothesis_transform=_WER_TRANSFORM,
            ),
            4,
        )
    except Exception:
        return None


def compute_cer(reference: str, hypothesis: str) -> Optional[float]:
    if not reference.strip() or not hypothesis.strip():
        return None
    try:
        return round(jiwer.cer(reference, hypothesis), 4)
    except Exception:
        return None


def compute_mer(reference: str, hypothesis: str) -> Optional[float]:
    if not reference.strip() or not hypothesis.strip():
        return None
    try:
        return round(
            jiwer.mer(
                reference,
                hypothesis,
                reference_transform=_WER_TRANSFORM,
                hypothesis_transform=_WER_TRANSFORM,
            ),
            4,
        )
    except Exception:
        return None


def get_openai_client() -> OpenAI:
    global openai_client
    if openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Set environment variable or use --skip-evaluation."
            )
        openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return openai_client


def evaluate_one(
    audio_id: str,
    variation: str,
    language: str,
    model_key: str,
    reference: str,
    hypothesis: str,
    wer: Optional[float],
    evaluator_model: str,
) -> dict[str, Any]:
    """Call GPT to evaluate a single (reference, hypothesis) pair."""
    if not hypothesis.strip():
        return _EMPTY_EVAL.copy()

    wer_str = f"{wer:.1%}" if wer is not None else "N/A (non-English)"
    user_msg = f"""\
Variation type  : {variation}
Language        : {language}
WER             : {wer_str}

REFERENCE TRANSCRIPT:
{reference}

MODEL HYPOTHESIS:
{hypothesis}

Evaluate the model's transcription quality against the reference."""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=evaluator_model,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content.strip())
        logger.info(
            f"  ✓ [{model_key}/{audio_id}] eval score: {result.get('overall_score', '?')}/10"
        )
        return result
    except json.JSONDecodeError as exc:
        logger.error(f"  JSON decode error in eval for {model_key}/{audio_id}: {exc}")
        return {"error": "json_decode_error"}
    except Exception as exc:
        logger.error(f"  Eval API error for {model_key}/{audio_id}: {exc}")
        return {"error": str(exc)}


def safe_avg(values: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def normalize_report_path(path: Path) -> Path:
    p = path.expanduser()
    if p.is_absolute():
        return p
    return Path.cwd() / p


def normalize_audio_path(audio_file: str, root_dir: Path) -> Path:
    normalized = audio_file.replace("\\", os.sep).replace("/", os.sep)
    p = Path(normalized)
    if p.is_absolute():
        return p
    return root_dir / p


def build_transcript_path(audio_path: Path, suffix: str) -> Path:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return audio_path.with_suffix(suffix)


def placeholder_evaluation(has_transcript: bool, message: str) -> dict[str, Any]:
    if has_transcript:
        return {
            "accuracy_score": None,
            "fluency_score": None,
            "completeness_score": None,
            "overall_score": None,
            "failure_summary": message,
            "error_categories": [],
            "improvement_suggestions": (
                "Run evaluator separately if you want GPT evaluation scores for Whisper."
            ),
            "notable_errors": [],
        }

    return {
        "accuracy_score": None,
        "fluency_score": None,
        "completeness_score": None,
        "overall_score": None,
        "failure_summary": message,
        "error_categories": ["deletion_errors"],
        "improvement_suggestions": (
            "Generate missing Whisper transcripts and rerun this updater script."
        ),
        "notable_errors": ["Whisper transcript file missing or empty"],
    }


def compute_whisper_aggregate(results: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    wers: list[Optional[float]] = []
    cers: list[Optional[float]] = []
    mers: list[Optional[float]] = []
    latencies: list[Optional[float]] = []
    scores: list[Optional[float]] = []
    per_var_wers: dict[str, list[Optional[float]]] = {}

    for result in results:
        variation = str(result.get("variation", "unknown"))
        per_var_wers.setdefault(variation, [])

        out = result.get("model_outputs", {}).get(model_key, {})
        evl = result.get("evaluations", {}).get(model_key, {})

        wer = out.get("wer")
        cer = out.get("cer")
        mer = out.get("mer")
        latency = out.get("latency_ms")
        score = evl.get("overall_score")

        if wer is not None:
            per_var_wers[variation].append(wer)
            wers.append(wer)
        if cer is not None:
            cers.append(cer)
        if mer is not None:
            mers.append(mer)
        if latency is not None and latency > 0:
            latencies.append(latency)
        if score is not None:
            scores.append(score)

    return {
        "avg_wer": safe_avg(wers),
        "avg_cer": safe_avg(cers),
        "avg_mer": safe_avg(mers),
        "avg_latency_ms": safe_avg(latencies),
        "avg_eval_score": safe_avg(scores),
        "sample_count": len(wers),
        "per_variation_avg_wer": {
            variation: safe_avg(variation_wers)
            for variation, variation_wers in per_var_wers.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject Whisper transcripts into benchmark_report.json"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmark_report.json"),
        help="Path to existing benchmark report JSON",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Root directory used to resolve relative audio_file paths (default: report directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. If omitted, updates --report in place",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="whisper",
        help="Model key to write under model_outputs",
    )
    parser.add_argument(
        "--model-display",
        type=str,
        default="Whisper",
        help="Display name to write in benchmark_metadata.transcription_models",
    )
    parser.add_argument(
        "--transcript-suffix",
        type=str,
        default=".whisper.txt",
        help="Suffix used for transcript file names",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=0.0,
        help="Latency to store for Whisper entries when unknown",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Exit with non-zero status if any transcript files are missing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and report stats without writing output JSON",
    )
    parser.add_argument(
        "--enable-evaluation",
        action="store_true",
        help="Call GPT to evaluate Whisper transcripts (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default=DEFAULT_EVALUATOR_MODEL,
        help="GPT model to use for evaluation (default: gpt-5.5)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="OpenAI base URL (default: OPENAI_BASE_URL env var or OpenAI default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.enable_evaluation:
        if args.openai_api_key:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key
        if args.openai_base_url:
            os.environ["OPENAI_BASE_URL"] = args.openai_base_url
        if not OPENAI_API_KEY:
            logger.error("Evaluation enabled but OPENAI_API_KEY not set.")
            return 1
        logger.info(f"Evaluation enabled using model: {args.evaluator_model}")

    report_path = normalize_report_path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    output_path = normalize_report_path(args.output) if args.output else report_path

    root_dir = (
        normalize_report_path(args.workspace_root)
        if args.workspace_root is not None
        else report_path.parent
    )

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("Invalid report: expected 'results' to be a list")

    missing_files: list[str] = []
    empty_files: list[str] = []
    updated_count = 0

    for item in results:
        if not isinstance(item, dict):
            continue

        audio_file = item.get("audio_file")
        reference = str(item.get("reference_transcript", ""))
        language = str(item.get("language", "en"))

        if not isinstance(audio_file, str) or not audio_file.strip():
            continue

        audio_path = normalize_audio_path(audio_file, root_dir)
        transcript_path = build_transcript_path(audio_path, args.transcript_suffix)

        transcript = ""
        source = "missing_file"
        error: Optional[str] = None

        if transcript_path.exists():
            transcript = transcript_path.read_text(encoding="utf-8").strip()
            if transcript:
                source = f"local_file:{transcript_path}"
            else:
                error = f"Whisper transcript file is empty: {transcript_path}"
                source = "empty_file"
                empty_files.append(str(transcript_path))
        else:
            error = f"Whisper transcript file not found: {transcript_path}"
            missing_files.append(str(transcript_path))

        if language == "en":
            wer = compute_wer(reference, transcript)
            cer = compute_cer(reference, transcript)
            mer = compute_mer(reference, transcript)
        else:
            wer = None
            cer = compute_cer(reference, transcript) if transcript else None
            mer = None

        model_outputs = item.setdefault("model_outputs", {})
        evaluations = item.setdefault("evaluations", {})

        model_outputs[args.model_key] = {
            "transcript": transcript,
            "wer": wer,
            "cer": cer,
            "mer": mer,
            "latency_ms": round(args.latency_ms, 1),
            "error": error,
            "transcription_source": source,
        }

        if transcript:
            if args.enable_evaluation:
                audio_id = item.get("audio_id", "unknown")
                variation = item.get("variation", "unknown")
                logger.info(f"  Evaluating [{audio_id}] with {args.evaluator_model}...")
                eval_result = evaluate_one(
                    audio_id=audio_id,
                    variation=variation,
                    language=language,
                    model_key=args.model_key,
                    reference=reference,
                    hypothesis=transcript,
                    wer=wer,
                    evaluator_model=args.evaluator_model,
                )
                evaluations[args.model_key] = eval_result
            else:
                evaluations[args.model_key] = placeholder_evaluation(
                    has_transcript=True,
                    message=(
                        "Whisper transcript added post-benchmark; GPT evaluation was not run. "
                        "Use --enable-evaluation to run GPT evaluation."
                    ),
                )
            updated_count += 1
        else:
            evaluations[args.model_key] = placeholder_evaluation(
                has_transcript=False,
                message=error or "Whisper transcript missing.",
            )

    metadata = report.setdefault("benchmark_metadata", {})
    transcription_models = metadata.setdefault("transcription_models", {})
    if isinstance(transcription_models, dict):
        transcription_models[args.model_key] = args.model_display

    aggregate_metrics = report.setdefault("aggregate_metrics", {})
    if isinstance(aggregate_metrics, dict):
        aggregate_metrics[args.model_key] = compute_whisper_aggregate(
            results,
            args.model_key,
        )

    print(f"Processed samples: {len(results)}")
    print(f"Whisper transcripts injected: {updated_count}")
    print(f"Missing transcript files: {len(missing_files)}")
    print(f"Empty transcript files: {len(empty_files)}")

    if args.dry_run:
        print("Dry run enabled: no file written.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Updated report written to: {output_path}")

    if args.strict_missing and (missing_files or empty_files):
        print("Strict mode enabled and missing/empty transcripts were found.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
