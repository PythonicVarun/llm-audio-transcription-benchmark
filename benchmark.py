#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         Multi-Model ASR Transcription Benchmark                  ║
║  Models : Gemma-4 | Gemini 3.1 Pro | Gemini 3 Flash | GPT-4o Tx  ║
║  Eval   : GPT-5.5                                                ║
║  Output : benchmark_report.json                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import base64
import json
import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Optional

import jiwer
from google import genai
from google.genai import types
from google.genai import types as genai_types
from openai import OpenAI
from tqdm import tqdm

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ⚠️  FILL IN YOUR API KEYS BELOW

GOOGLE_API_KEY = os.getenv("CODEX_STRAIVE_OPENAI_TOKEN", "YOUR_GOOGLE_API_KEY")  # google-genai key
OPENAI_API_KEY = os.getenv("CODEX_STRAIVE_OPENAI_TOKEN", "YOUR_OPENAI_API_KEY")  # openai key
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://llmfoundry.straivedemo.com/openai/v1").strip() or None  # e.g. https://api.openai.com/v1

DEFAULT_MANUAL_TRANSCRIPT_DIR = pathlib.Path(
    os.getenv("BENCHMARK_MANUAL_TRANSCRIPT_DIR", "manual_ai_studio_transcripts")
)
DEFAULT_GOOGLE_MANUAL_MODE = (
    os.getenv("BENCHMARK_GOOGLE_MANUAL_MODE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_OPENAI_MANUAL_MODE = (
    os.getenv("BENCHMARK_OPENAI_MANUAL_MODE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

# MODEL REGISTRY

TRANSCRIPTION_MODELS: dict[str, dict] = {
    "gemma-4": {
        "provider": "google",
        "model_id": "gemma-4-31b-it",
        "display": "Gemma-4",
    },
    "gemini-3.1-pro-preview": {
        "provider": "google",
        "model_id": "gemini-3.1-pro-preview",
        "display": "Gemini 3.1 Pro Preview",
    },
    "gemini-3-flash-preview": {
        "provider": "google",
        "model_id": "gemini-3-flash-preview",
        "display": "Gemini 3 Flash Preview",
    },
    "gpt-4o-transcribe": {
        "provider": "openai",
        "model_id": "gpt-4o-transcribe",
        "display": "GPT-4o Transcribe",
    },
}

EVALUATOR_MODEL = "gpt-5.5"
MANIFEST_PATH = pathlib.Path("manifest.json")
OUTPUT_PATH = pathlib.Path("benchmark_report.json")
AUDIO_DIR = pathlib.Path("audio_files")

# AUDIO MIME MAP

MIME_MAP: dict[str, str] = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
}

# API CLIENTS

google_client: Optional[genai.Client] = None
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def get_google_client() -> genai.Client:
    global google_client
    if google_client is None:
        if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("YOUR_"):
            raise RuntimeError(
                "GOOGLE_API_KEY is not configured. Set it or run with --google-manual-mode."
            )
        google_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(
                base_url="https://llmfoundry.straivedemo.com/gemini/",
            )
        )
    return google_client

# TRANSCRIPTION LOGIC

TRANSCRIPTION_PROMPT = (
    "You are a professional transcriptionist. "
    "Transcribe every word spoken in this audio with maximum accuracy. "
    "Output ONLY the verbatim transcription - no speaker labels, no timestamps, "
    "no explanations, no formatting. "
    "If the audio is non-English, transcribe in the original language as-is."
)

MULTI_SPEAKER_TRANSCRIPTION_PROMPT = (
    "You are a professional transcriptionist. "
    "Transcribe every word spoken in this multi-speaker audio with maximum accuracy. "
    "Preserve speaker turns using labels in the exact format SPEAKER_1:, SPEAKER_2:, "
    "and so on. Put each speaker turn on its own line, keep labels consistent across "
    "the audio, and output only the labeled transcript - no timestamps, explanations, "
    "or extra formatting."
)


def transcription_prompt_for_variation(variation: str) -> str:
    if variation == "multiple_speakers":
        return MULTI_SPEAKER_TRANSCRIPTION_PROMPT
    return TRANSCRIPTION_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ASR benchmark. Use --google-manual-mode to read Google model "
            "transcripts from local text files exported from AI Studio."
        )
    )
    parser.add_argument(
        "--google-manual-mode",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GOOGLE_MANUAL_MODE,
        help=(
            "Enable or disable manual mode for Google models. When enabled, "
            "transcripts are read from files instead of Google API calls."
        ),
    )
    parser.add_argument(
        "--openai-manual-mode",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OPENAI_MANUAL_MODE,
        help=(
            "Enable or disable manual mode for OpenAI models. Use this if the "
            "OpenAI-compatible endpoint does not expose audio transcription."
        ),
    )
    parser.add_argument(
        "--manual-transcript-dir",
        type=pathlib.Path,
        default=DEFAULT_MANUAL_TRANSCRIPT_DIR,
        help=(
            "Directory containing manual transcripts. Expected files: "
            "<dir>/<model_key>/<audio_id>.txt"
        ),
    )
    return parser.parse_args()


def load_manual_transcript(
    model_key: str,
    audio_id: str,
    audio_path: pathlib.Path,
    manual_transcript_dir: pathlib.Path,
) -> tuple[str, pathlib.Path]:
    candidates = [
        manual_transcript_dir / model_key / f"{audio_id}.txt",
        manual_transcript_dir / model_key / f"{audio_path.stem}.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            transcript = candidate.read_text(encoding="utf-8").strip()
            if transcript:
                return transcript, candidate
            raise ValueError(f"Manual transcript file is empty: {candidate}")

    expected = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Manual transcript not found for {model_key}/{audio_id}. Expected one of:\n  {expected}"
    )


def transcribe_google(
    model_id: str,
    audio_path: pathlib.Path,
    prompt: str = TRANSCRIPTION_PROMPT,
) -> tuple[str, float]:
    """
    Transcribe using google-genai (Gemma-4 / Gemini models).
    Uses inline bytes for files < ~20 MB; falls back to Files API otherwise.
    """
    mime = MIME_MAP.get(audio_path.suffix.lower(), "audio/wav")
    audio_bytes = audio_path.read_bytes()
    client = get_google_client()

    t0 = time.monotonic()

    if len(audio_bytes) <= 20 * 1024 * 1024:
        # Inline audio (small files)
        response = client.models.generate_content(
            model=model_id,
            contents=[
                genai_types.Content(
                    parts=[
                        genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                        genai_types.Part.from_text(text=prompt),
                    ]
                )
            ],
        )
    else:
        # Files API (large files)
        uploaded = client.files.upload(
            path=audio_path,
            config=genai_types.UploadFileConfig(mime_type=mime),
        )
        # Poll until file is active
        for _ in range(30):
            status = client.files.get(name=uploaded.name)
            if status.state.name == "ACTIVE":
                break
            time.sleep(2)

        response = client.models.generate_content(
            model=model_id,
            contents=[
                genai_types.Content(
                    parts=[
                        genai_types.Part.from_uri(
                            file_uri=uploaded.uri, mime_type=mime
                        ),
                        genai_types.Part.from_text(text=prompt),
                    ]
                )
            ],
        )

    latency_ms = (time.monotonic() - t0) * 1000
    transcript = (response.text or "").strip()
    return transcript, latency_ms


def transcribe_openai(
    model_id: str,
    audio_path: pathlib.Path,
    prompt: str = TRANSCRIPTION_PROMPT,
) -> tuple[str, float]:
    """
    Transcribe using OpenAI-compatible audio transcriptions.
    """
    t0 = time.monotonic()
    with audio_path.open("rb") as audio_file:
        response = openai_client.audio.transcriptions.create(
            model=model_id,
            file=audio_file,
            prompt=prompt,
        )
    latency_ms = (time.monotonic() - t0) * 1000
    if isinstance(response, dict):
        transcript = response.get("text", "")
    else:
        transcript = getattr(response, "text", response)
    transcript = str(transcript or "").strip()
    return transcript, latency_ms


def transcribe(
    model_key: str,
    audio_id: str,
    audio_path: pathlib.Path,
    variation: str = "",
    google_manual_mode: bool = False,
    openai_manual_mode: bool = False,
    manual_transcript_dir: pathlib.Path = DEFAULT_MANUAL_TRANSCRIPT_DIR,
) -> tuple[str, float, Optional[str], str]:
    """
    Dispatch to the correct provider and return (transcript, latency_ms, error, source).
    """
    cfg = TRANSCRIPTION_MODELS[model_key]
    prompt = transcription_prompt_for_variation(variation)
    try:
        if (
            (cfg["provider"] == "google" and google_manual_mode)
            or (cfg["provider"] == "openai" and openai_manual_mode)
        ):
            transcript, source_file = load_manual_transcript(
                model_key=model_key,
                audio_id=audio_id,
                audio_path=audio_path,
                manual_transcript_dir=manual_transcript_dir,
            )
            return transcript, 0.0, None, f"manual_file:{source_file}"

        if cfg["provider"] == "google":
            t, lat = transcribe_google(cfg["model_id"], audio_path, prompt)
        else:
            t, lat = transcribe_openai(cfg["model_id"], audio_path, prompt)
        return t, lat, None, "api"
    except Exception as exc:
        logger.error(f"  ✗ [{model_key}] {audio_path.name} → {exc}")
        return "", 0.0, str(exc), "error"


# METRICS

_wer_transform = jiwer.Compose(
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
                reference_transform=_wer_transform,
                hypothesis_transform=_wer_transform,
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
    """Match Error Rate — useful for multi-speaker / noisy audio."""
    if not reference.strip() or not hypothesis.strip():
        return None
    try:
        return round(
            jiwer.mer(
                reference,
                hypothesis,
                reference_transform=_wer_transform,
                hypothesis_transform=_wer_transform,
            ),
            4,
        )
    except Exception:
        return None


# PER-SAMPLE EVALUATION (GPT-5.4)

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


def evaluate_one(
    audio_id: str,
    variation: str,
    language: str,
    model_key: str,
    reference: str,
    hypothesis: str,
    wer: Optional[float],
) -> dict:
    """Call GPT-5.4 to evaluate a single (reference, hypothesis) pair."""
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
        resp = openai_client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            # max_completion_tokens=700,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content.strip())
    except json.JSONDecodeError as exc:
        logger.error(f"  JSON decode error in eval for {model_key}/{audio_id}: {exc}")
        return {"error": "json_decode_error", "raw": resp.choices[0].message.content}
    except Exception as exc:
        logger.error(f"  Eval API error for {model_key}/{audio_id}: {exc}")
        return {"error": str(exc)}


# GLOBAL SUMMARY (GPT-5.4)

_SUMMARY_SYSTEM = """\
You are a senior ML research engineer specialising in speech recognition evaluation.
You will receive aggregated benchmark results and must produce a comprehensive analysis report as JSON.
Respond ONLY with a valid JSON object — no markdown, no extra text.
"""

_SUMMARY_SCHEMA = """\
{
  "overall_ranking": [
    {"rank": 1, "model": "<key>", "avg_wer": 0.0, "avg_score": 0.0, "verdict": "<short>"}
  ],
  "per_variation_winner": {
    "clean": "<model_key>",
    "background_noise": "<model_key>",
    "multiple_speakers": "<model_key>",
    "accents": "<model_key>",
    "multilingual": "<model_key>"
  },
  "model_analysis": {
    "<model_key>": {
      "avg_wer":             0.0,
      "avg_cer":             0.0,
      "avg_overall_score":   0.0,
      "avg_latency_ms":      0.0,
      "best_variation":      "<variation>",
      "worst_variation":     "<variation>",
      "top_error_categories": ["<category>"],
      "strengths":           ["<strength>"],
      "weaknesses":          ["<weakness>"],
      "failure_patterns":    "<detailed paragraph>",
      "how_to_improve":      "<detailed paragraph with concrete steps>",
      "overall_verdict":     "<two to three sentence conclusion>"
    }
  },
  "benchmark_insights": ["<key insight>"],
  "dataset_difficulty_ranking": ["<variation> (easiest)", "...", "<variation> (hardest)"],
  "recommendations": ["<actionable recommendation>"]
}
"""


def generate_summary(all_results: list[dict]) -> dict:
    """Ask GPT-5.4 to produce the final benchmark summary."""

    # Build a condensed payload (no base64 audio)
    condensed = []
    for r in all_results:
        condensed.append(
            {
                "audio_id": r["audio_id"],
                "variation": r["variation"],
                "dataset": r["dataset"],
                "language": r.get("language", "en"),
                "model_metrics": {
                    mk: {
                        "wer": r["model_outputs"].get(mk, {}).get("wer"),
                        "cer": r["model_outputs"].get(mk, {}).get("cer"),
                        "latency_ms": r["model_outputs"].get(mk, {}).get("latency_ms"),
                        "overall_score": r["evaluations"]
                        .get(mk, {})
                        .get("overall_score"),
                        "accuracy_score": r["evaluations"]
                        .get(mk, {})
                        .get("accuracy_score"),
                        "failure_summary": r["evaluations"]
                        .get(mk, {})
                        .get("failure_summary"),
                        "error_categories": r["evaluations"]
                        .get(mk, {})
                        .get("error_categories"),
                    }
                    for mk in TRANSCRIPTION_MODELS
                },
            }
        )

    user_msg = (
        f"Here are transcription benchmark results for {len(all_results)} audio samples "
        f"(5 variations × 2 files) across 4 models.\n\n"
        f"Model keys: {list(TRANSCRIPTION_MODELS.keys())}\n\n"
        f"Results:\n{json.dumps(condensed, indent=2)}\n\n"
        f"Produce a comprehensive summary following this exact JSON schema:\n{_SUMMARY_SCHEMA}"
    )

    try:
        resp = openai_client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            # max_completion_tokens=3000,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as exc:
        logger.error(f"Summary generation failed: {exc}")
        return {"error": str(exc)}


# AGGREGATE STATS


def _safe_avg(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def compute_aggregates(all_results: list[dict]) -> dict:
    stats: dict[str, dict] = {
        mk: {
            "wers": [],
            "cers": [],
            "mers": [],
            "latencies": [],
            "scores": [],
        }
        for mk in TRANSCRIPTION_MODELS
    }

    variation_wers: dict[str, dict[str, list]] = {}

    for r in all_results:
        var = r["variation"]
        if var not in variation_wers:
            variation_wers[var] = {mk: [] for mk in TRANSCRIPTION_MODELS}

        for mk in TRANSCRIPTION_MODELS:
            out = r["model_outputs"].get(mk, {})
            evl = r["evaluations"].get(mk, {})

            if (w := out.get("wer")) is not None:
                stats[mk]["wers"].append(w)
                variation_wers[var][mk].append(w)
            if (c := out.get("cer")) is not None:
                stats[mk]["cers"].append(c)
            if (m := out.get("mer")) is not None:
                stats[mk]["mers"].append(m)
            if (lat := out.get("latency_ms")) is not None and lat > 0:
                stats[mk]["latencies"].append(lat)
            if (s := evl.get("overall_score")) is not None:
                stats[mk]["scores"].append(s)

    aggregate: dict[str, dict] = {}
    for mk, s in stats.items():
        aggregate[mk] = {
            "avg_wer": _safe_avg(s["wers"]),
            "avg_cer": _safe_avg(s["cers"]),
            "avg_mer": _safe_avg(s["mers"]),
            "avg_latency_ms": _safe_avg(s["latencies"]),
            "avg_eval_score": _safe_avg(s["scores"]),
            "sample_count": len(s["wers"]),
            "per_variation_avg_wer": {
                var: _safe_avg(wers[mk]) for var, wers in variation_wers.items()
            },
        }
    return aggregate


# MAIN RUNNER


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest.json not found at {MANIFEST_PATH.resolve()}")
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(
    google_manual_mode: bool = False,
    openai_manual_mode: bool = False,
    manual_transcript_dir: pathlib.Path = DEFAULT_MANUAL_TRANSCRIPT_DIR,
) -> None:
    manual_transcript_dir = pathlib.Path(manual_transcript_dir)
    manifest = load_manifest()
    logger.info(f"Loaded {len(manifest)} audio samples from manifest.json")
    if google_manual_mode:
        logger.info(
            "Google manual mode enabled. Reading manual transcripts from: %s",
            manual_transcript_dir.resolve(),
        )
    if openai_manual_mode:
        logger.info(
            "OpenAI manual mode enabled. Reading manual transcripts from: %s",
            manual_transcript_dir.resolve(),
        )

    # Verify all audio files exist before starting
    missing = [e["file"] for e in manifest if not pathlib.Path(e["file"]).exists()]
    if missing:
        logger.warning(
            f"Missing audio files (will be skipped):\n  " + "\n  ".join(missing)
        )

    all_results: list[dict] = []

    for entry in tqdm(manifest, desc="Audio samples", unit="file"):
        audio_path = pathlib.Path(entry["file"])
        if not audio_path.exists():
            logger.warning(f"Skipping {entry['id']} — file not found: {audio_path}")
            continue

        audio_id = entry["id"]
        variation = entry["variation"]
        reference = entry["reference"]
        language = entry.get("language", "en")

        logger.info(f"\n{'─'*65}")
        logger.info(f"▶ [{audio_id}]  {entry['dataset']}  ({entry['variation_label']})")

        # Step 1: Transcribe with all 4 models
        model_outputs: dict[str, dict] = {}

        for model_key in TRANSCRIPTION_MODELS:
            if (
                (
                    google_manual_mode
                    and TRANSCRIPTION_MODELS[model_key]["provider"] == "google"
                )
                or (
                    openai_manual_mode
                    and TRANSCRIPTION_MODELS[model_key]["provider"] == "openai"
                )
            ):
                logger.info(
                    f"  ▷ Loading manual transcript for {TRANSCRIPTION_MODELS[model_key]['display']} ..."
                )
            else:
                logger.info(
                    f"  ▷ Transcribing with {TRANSCRIPTION_MODELS[model_key]['display']} ..."
                )

            transcript, latency_ms, error, source = transcribe(
                model_key=model_key,
                audio_id=audio_id,
                audio_path=audio_path,
                variation=variation,
                google_manual_mode=google_manual_mode,
                openai_manual_mode=openai_manual_mode,
                manual_transcript_dir=manual_transcript_dir,
            )

            # Compute metrics only for English (WER is language-dependent)
            if language == "en":
                wer = compute_wer(reference, transcript)
                cer = compute_cer(reference, transcript)
                mer = compute_mer(reference, transcript)
            else:
                # For non-English: CER is still meaningful; WER less so without tokenizer
                wer = None
                cer = compute_cer(reference, transcript) if transcript else None
                mer = None

            model_outputs[model_key] = {
                "transcript": transcript,
                "wer": wer,
                "cer": cer,
                "mer": mer,
                "latency_ms": round(latency_ms, 1),
                "error": error,
                "transcription_source": source,
            }

            wer_str = f"WER={wer:.1%}" if wer is not None else "WER=N/A"
            logger.info(
                f"    {wer_str}  CER={cer:.1%}  latency={latency_ms:.0f}ms"
                if cer is not None
                else f"    {wer_str}  latency={latency_ms:.0f}ms"
            )

        # Step 2: Evaluate with GPT-5.4
        logger.info(f"  ▷ Evaluating all outputs with {EVALUATOR_MODEL} ...")
        evaluations: dict[str, dict] = {}

        for model_key in TRANSCRIPTION_MODELS:
            out = model_outputs[model_key]
            evaluations[model_key] = evaluate_one(
                audio_id=audio_id,
                variation=variation,
                language=language,
                model_key=model_key,
                reference=reference,
                hypothesis=out["transcript"],
                wer=out["wer"],
            )
            score = evaluations[model_key].get("overall_score", "?")
            logger.info(
                f"    {TRANSCRIPTION_MODELS[model_key]['display']:30s} → score={score}/10"
            )

        # Build result record
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

        all_results.append(result)

    # Step 3: Global summary
    logger.info(f"\n{'─'*65}")
    logger.info(f"▶ Generating benchmark summary with {EVALUATOR_MODEL} ...")
    summary = generate_summary(all_results)
    aggregates = compute_aggregates(all_results)

    # Assemble final report
    report = {
        "benchmark_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluator_model": EVALUATOR_MODEL,
            "google_manual_mode": google_manual_mode,
            "openai_manual_mode": openai_manual_mode,
            "manual_transcript_dir": str(manual_transcript_dir),
            "transcription_models": {
                mk: cfg["display"] for mk, cfg in TRANSCRIPTION_MODELS.items()
            },
            "total_audio_samples": len(all_results),
            "dataset_variations": [
                "clean",
                "background_noise",
                "multiple_speakers",
                "accents",
                "multilingual",
            ],
            "samples_per_variation": 2,
            "metrics_used": ["WER", "CER", "MER", "GPT-5.4 eval score (1-10)"],
            "note_on_multilingual": (
                "WER is omitted for non-English samples; CER is reported instead. "
                "GPT-5.4 evaluation judges quality holistically for all languages."
            ),
        },
        "aggregate_metrics": aggregates,
        "results": all_results,
        "summary": summary,
    }

    # Write JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Console summary
    logger.info(f"\n{'═'*65}")
    logger.info(f"✅  Benchmark complete → {OUTPUT_PATH.resolve()}")
    logger.info(f"    Samples processed : {len(all_results)}")
    logger.info(f"    {'Model':<35} {'Avg WER':>9}  {'Avg Score':>10}  {'Avg Lat':>10}")
    logger.info(f"    {'─'*35} {'─'*9}  {'─'*10}  {'─'*10}")
    for mk, agg in aggregates.items():
        wer_s = f"{agg['avg_wer']:.1%}" if agg["avg_wer"] is not None else "N/A"
        score_s = (
            f"{agg['avg_eval_score']:.1f}/10"
            if agg["avg_eval_score"] is not None
            else "N/A"
        )
        lat_s = (
            f"{agg['avg_latency_ms']:.0f}ms"
            if agg["avg_latency_ms"] is not None
            else "N/A"
        )
        logger.info(
            f"    {TRANSCRIPTION_MODELS[mk]['display']:<35} {wer_s:>9}  {score_s:>10}  {lat_s:>10}"
        )
    logger.info(f"{'═'*65}\n")


# ENTRY POINT

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        google_manual_mode=args.google_manual_mode,
        openai_manual_mode=args.openai_manual_mode,
        manual_transcript_dir=args.manual_transcript_dir,
    )
