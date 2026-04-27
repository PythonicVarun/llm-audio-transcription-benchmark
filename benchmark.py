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
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Optional

import jiwer
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
DEFAULT_LOCAL_MANUAL_MODE = (
    os.getenv("BENCHMARK_LOCAL_MANUAL_MODE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_LOCAL_OPENAI_BASE_URL = (
    os.getenv("BENCHMARK_LOCAL_OPENAI_BASE_URL")
    or os.getenv("LOCAL_OPENAI_BASE_URL")
    or "http://localhost:8000/v1"
).strip()
DEFAULT_LOCAL_OPENAI_API_KEY = (
    os.getenv("BENCHMARK_LOCAL_OPENAI_API_KEY")
    or os.getenv("LOCAL_OPENAI_API_KEY")
    or "local"
).strip()
DEFAULT_LOCAL_AUDIO_MODE = os.getenv("BENCHMARK_LOCAL_AUDIO_MODE", "input_audio").strip().lower()
DEFAULT_LOCAL_MODELS_RAW = os.getenv("BENCHMARK_LOCAL_MODELS", "").strip()
DEFAULT_LOCAL_COMMAND_MODELS_RAW = os.getenv("BENCHMARK_LOCAL_COMMAND_MODELS", "").strip()
DEFAULT_LOCAL_COMMAND_TIMEOUT_SECONDS = float(
    os.getenv("BENCHMARK_LOCAL_COMMAND_TIMEOUT_SECONDS", "600")
)
DEFAULT_MODEL_FILTER_RAW = os.getenv("BENCHMARK_MODELS", "").strip()
LOCAL_AUDIO_MODES = {"input_audio", "audio_url"}

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

OPENAI_AUDIO_FORMAT_MAP: dict[str, str] = {
    ".wav": "wav",
    ".flac": "flac",
    ".mp3": "mp3",
    ".ogg": "ogg",
    ".m4a": "mp4",
    ".opus": "opus",
    ".webm": "webm",
}


def _split_specs(raw: str) -> list[str]:
    if not raw:
        return []
    if raw.lstrip().startswith(("{", "[")):
        return [raw]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _slugify_model_key(value: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._")
    return (key or "local-model").lower()


def _display_name_from_key(model_key: str) -> str:
    return model_key.replace("_", " ").replace("-", " ").title()


def _env_or_value(value: Optional[str], default: str) -> str:
    if not value:
        return default
    if value.startswith("env:"):
        return os.getenv(value[4:], default)
    return value


def _build_local_openai_model_cfg(
    key: str,
    spec: Any,
    default_base_url: str,
    default_api_key: str,
    default_audio_mode: str,
) -> tuple[str, dict]:
    if isinstance(spec, str):
        model_id = spec.strip()
        display = _display_name_from_key(key)
        base_url = default_base_url
        api_key = default_api_key
        audio_mode = default_audio_mode
    elif isinstance(spec, dict):
        model_id = str(spec.get("model_id") or spec.get("model") or "").strip()
        display = str(spec.get("display") or _display_name_from_key(key)).strip()
        base_url = str(spec.get("base_url") or default_base_url).strip()
        if spec.get("api_key_env") and not spec.get("api_key"):
            api_key = os.getenv(str(spec["api_key_env"]), default_api_key)
        else:
            api_key = _env_or_value(spec.get("api_key"), default_api_key)
        audio_mode = str(spec.get("audio_mode") or default_audio_mode).strip().lower()
    else:
        raise ValueError(f"Unsupported local model config for {key!r}: {spec!r}")

    if not model_id:
        raise ValueError(f"Local model {key!r} is missing a model_id")
    if not base_url:
        raise ValueError(f"Local model {key!r} is missing a base_url")
    if audio_mode not in LOCAL_AUDIO_MODES:
        raise ValueError(
            f"Local model {key!r} has unsupported audio_mode={audio_mode!r}. "
            f"Use one of: {', '.join(sorted(LOCAL_AUDIO_MODES))}"
        )

    return key, {
        "provider": "local_openai_chat",
        "model_id": model_id,
        "display": display,
        "base_url": base_url,
        "api_key": api_key or "local",
        "audio_mode": audio_mode,
    }


def parse_local_openai_model_specs(
    specs: list[str],
    default_base_url: str = DEFAULT_LOCAL_OPENAI_BASE_URL,
    default_api_key: str = DEFAULT_LOCAL_OPENAI_API_KEY,
    default_audio_mode: str = DEFAULT_LOCAL_AUDIO_MODE,
) -> dict[str, dict]:
    """
    Parse audio-capable local OpenAI-compatible chat models.

    Compact form:
      --local-model gemma-4-local=gemma-4

    JSON env form:
      BENCHMARK_LOCAL_MODELS='{"gemma-4-local":{"model_id":"gemma-4","base_url":"http://localhost:8000/v1"}}'
    """
    models: dict[str, dict] = {}
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue

        if spec.lstrip().startswith(("{", "[")):
            payload = json.loads(spec)
            if isinstance(payload, dict):
                items = payload.items()
            elif isinstance(payload, list):
                items = []
                for item in payload:
                    if not isinstance(item, dict):
                        raise ValueError("JSON local model lists must contain objects")
                    key = str(
                        item.get("key")
                        or item.get("name")
                        or _slugify_model_key(
                            str(item.get("model_id") or item.get("model") or "")
                        )
                    )
                    items.append((key, item))
            else:
                raise ValueError("Local model JSON must be an object or list")

            for key, cfg_spec in items:
                key = str(key).strip()
                model_key, cfg = _build_local_openai_model_cfg(
                    key=key,
                    spec=cfg_spec,
                    default_base_url=default_base_url,
                    default_api_key=default_api_key,
                    default_audio_mode=default_audio_mode,
                )
                models[model_key] = cfg
            continue

        if "=" in spec:
            key, model_id = spec.split("=", 1)
            key = key.strip()
            model_id = model_id.strip()
        else:
            model_id = spec
            key = _slugify_model_key(model_id)
        model_key, cfg = _build_local_openai_model_cfg(
            key=key,
            spec=model_id,
            default_base_url=default_base_url,
            default_api_key=default_api_key,
            default_audio_mode=default_audio_mode,
        )
        models[model_key] = cfg
    return models


def _build_local_command_model_cfg(
    key: str,
    spec: Any,
    default_timeout_seconds: float,
) -> tuple[str, dict]:
    if isinstance(spec, str):
        payload = spec.strip()
        if "::" in payload:
            display, command = payload.split("::", 1)
            display = display.strip() or _display_name_from_key(key)
            command = command.strip()
        else:
            display = _display_name_from_key(key)
            command = payload
        cfg = {
            "provider": "local_command",
            "display": display,
            "command_template": command,
            "timeout_seconds": default_timeout_seconds,
        }
    elif isinstance(spec, dict):
        display = str(spec.get("display") or _display_name_from_key(key)).strip()
        command = str(spec.get("command") or spec.get("command_template") or "").strip()
        cfg = {
            "provider": "local_command",
            "display": display,
            "command_template": command,
            "cwd": spec.get("cwd"),
            "output_file_template": spec.get("output_file")
            or spec.get("output_file_template"),
            "timeout_seconds": float(
                spec.get("timeout_seconds", default_timeout_seconds)
            ),
        }
    else:
        raise ValueError(f"Unsupported local command model config for {key!r}: {spec!r}")

    if not cfg.get("command_template") and not cfg.get("output_file_template"):
        raise ValueError(
            f"Local command model {key!r} needs a command or output_file template"
        )
    return key, cfg


def parse_local_command_model_specs(
    specs: list[str],
    default_timeout_seconds: float = DEFAULT_LOCAL_COMMAND_TIMEOUT_SECONDS,
) -> dict[str, dict]:
    """
    Parse local command/file transcript models such as Parakeet.

    Compact form:
      --local-command-model parakeet=Parakeet Local::python parakeet_asr.py {audio_q}

    The command should print the transcript to stdout. JSON config can also
    provide output_file when the tool writes transcripts to disk.
    """
    models: dict[str, dict] = {}
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue

        if spec.lstrip().startswith(("{", "[")):
            payload = json.loads(spec)
            if isinstance(payload, dict):
                items = payload.items()
            elif isinstance(payload, list):
                items = []
                for item in payload:
                    if not isinstance(item, dict):
                        raise ValueError(
                            "JSON local command model lists must contain objects"
                        )
                    key = str(item.get("key") or item.get("name") or "").strip()
                    if not key:
                        raise ValueError("JSON local command model is missing key")
                    items.append((key, item))
            else:
                raise ValueError("Local command model JSON must be an object or list")

            for key, cfg_spec in items:
                model_key, cfg = _build_local_command_model_cfg(
                    str(key).strip(),
                    cfg_spec,
                    default_timeout_seconds,
                )
                models[model_key] = cfg
            continue

        if "=" not in spec:
            raise ValueError(
                "Local command model specs must use KEY=COMMAND or KEY=DISPLAY::COMMAND"
            )
        key, payload = spec.split("=", 1)
        model_key, cfg = _build_local_command_model_cfg(
            key.strip(),
            payload.strip(),
            default_timeout_seconds,
        )
        models[model_key] = cfg
    return models


def register_local_models(
    local_model_specs: Optional[list[str]] = None,
    local_command_model_specs: Optional[list[str]] = None,
    default_base_url: str = DEFAULT_LOCAL_OPENAI_BASE_URL,
    default_api_key: str = DEFAULT_LOCAL_OPENAI_API_KEY,
    default_audio_mode: str = DEFAULT_LOCAL_AUDIO_MODE,
    default_command_timeout_seconds: float = DEFAULT_LOCAL_COMMAND_TIMEOUT_SECONDS,
) -> None:
    local_models = parse_local_openai_model_specs(
        specs=local_model_specs or [],
        default_base_url=default_base_url,
        default_api_key=default_api_key,
        default_audio_mode=default_audio_mode,
    )
    local_models.update(
        parse_local_command_model_specs(
            local_command_model_specs or [],
            default_timeout_seconds=default_command_timeout_seconds,
        )
    )

    for model_key in local_models:
        if model_key in TRANSCRIPTION_MODELS:
            raise ValueError(f"Local model key {model_key!r} already exists")
    TRANSCRIPTION_MODELS.update(local_models)


# API CLIENTS

google_client: Optional[Any] = None
openai_client: Optional[Any] = None
local_openai_clients: dict[tuple[str, str], Any] = {}


def get_google_client() -> Any:
    global google_client
    if google_client is None:
        if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("YOUR_"):
            raise RuntimeError(
                "GOOGLE_API_KEY is not configured. Set it or run with --google-manual-mode."
            )
        from google import genai
        from google.genai import types as google_types

        google_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=google_types.HttpOptions(
                base_url="https://llmfoundry.straivedemo.com/gemini/",
            )
        )
    return google_client


def get_openai_client() -> Any:
    global openai_client
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return openai_client


def get_local_openai_client(base_url: str, api_key: str) -> Any:
    key = (base_url, api_key or "local")
    if key not in local_openai_clients:
        from openai import OpenAI

        local_openai_clients[key] = OpenAI(
            api_key=api_key or "local",
            base_url=base_url,
        )
    return local_openai_clients[key]

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
            "Run ASR benchmark. Supports cloud APIs, manual transcripts, "
            "local OpenAI-compatible audio chat models, and local ASR commands."
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
    parser.add_argument(
        "--local-model",
        action="append",
        default=_split_specs(DEFAULT_LOCAL_MODELS_RAW),
        metavar="KEY=MODEL_ID",
        help=(
            "Add an audio-capable local OpenAI-compatible chat model. Repeatable. "
            "Example: --local-model gemma-4-local=gemma-4"
        ),
    )
    parser.add_argument(
        "--local-openai-base-url",
        default=DEFAULT_LOCAL_OPENAI_BASE_URL,
        help=(
            "Base URL for local OpenAI-compatible servers "
            "(default: BENCHMARK_LOCAL_OPENAI_BASE_URL, LOCAL_OPENAI_BASE_URL, "
            "or http://localhost:8000/v1)."
        ),
    )
    parser.add_argument(
        "--local-openai-api-key",
        default=DEFAULT_LOCAL_OPENAI_API_KEY,
        help="API key for local OpenAI-compatible servers (default: local).",
    )
    parser.add_argument(
        "--local-audio-mode",
        choices=sorted(LOCAL_AUDIO_MODES),
        default=DEFAULT_LOCAL_AUDIO_MODE,
        help=(
            "Audio content block format for local chat models. Use input_audio "
            "for OpenAI-style servers, audio_url for vLLM-style servers."
        ),
    )
    parser.add_argument(
        "--local-command-model",
        action="append",
        default=_split_specs(DEFAULT_LOCAL_COMMAND_MODELS_RAW),
        metavar="KEY=DISPLAY::COMMAND",
        help=(
            "Add a local ASR command model such as Parakeet. The command should "
            "print a transcript to stdout. Placeholders include {audio}, "
            "{audio_q}, {audio_abs}, {audio_abs_q}, {audio_id}, {audio_stem}, "
            "{audio_name}, {audio_parent}, {prompt}, {prompt_q}."
        ),
    )
    parser.add_argument(
        "--local-command-timeout-seconds",
        type=float,
        default=DEFAULT_LOCAL_COMMAND_TIMEOUT_SECONDS,
        help="Default timeout for local command models.",
    )
    parser.add_argument(
        "--local-manual-mode",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOCAL_MANUAL_MODE,
        help=(
            "Enable or disable manual mode for local models. When enabled, "
            "transcripts are read from files instead of local model calls."
        ),
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODEL_FILTER_RAW,
        help=(
            "Optional comma-separated model keys to run after local models are "
            "registered. Example: --models parakeet,gemma-4-local"
        ),
    )
    parser.add_argument(
        "--append-report",
        action="store_true",
        help=(
            "If benchmark_report.json already exists, merge this run into it "
            "instead of replacing it. Existing audio rows are matched by audio_id; "
            "new model outputs/evaluations are added or updated."
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
    from google.genai import types as genai_types

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
    client = get_openai_client()
    t0 = time.monotonic()
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
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


def local_audio_content(
    audio_path: pathlib.Path,
    prompt: str,
    audio_mode: str,
) -> list[dict[str, Any]]:
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    suffix = audio_path.suffix.lower()
    mime = MIME_MAP.get(suffix, "audio/wav")

    if audio_mode == "input_audio":
        return [
            {"type": "text", "text": prompt},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": OPENAI_AUDIO_FORMAT_MAP.get(
                        suffix,
                        suffix.lstrip(".") or "wav",
                    ),
                },
            },
        ]

    if audio_mode == "audio_url":
        return [
            {"type": "text", "text": prompt},
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:{mime};base64,{audio_b64}"},
            },
        ]

    raise ValueError(
        f"Unsupported local audio mode {audio_mode!r}. "
        f"Use one of: {', '.join(sorted(LOCAL_AUDIO_MODES))}"
    )


def transcribe_local_openai_chat(
    model_id: str,
    audio_path: pathlib.Path,
    prompt: str = TRANSCRIPTION_PROMPT,
    base_url: str = DEFAULT_LOCAL_OPENAI_BASE_URL,
    api_key: str = DEFAULT_LOCAL_OPENAI_API_KEY,
    audio_mode: str = DEFAULT_LOCAL_AUDIO_MODE,
) -> tuple[str, float]:
    """
    Transcribe using a local OpenAI-compatible chat/completions endpoint.

    The local model must be audio-capable. Text-only local LLMs cannot
    transcribe raw audio unless the server adds audio preprocessing.
    """
    client = get_local_openai_client(base_url=base_url, api_key=api_key)
    content = local_audio_content(audio_path, prompt, audio_mode)

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        temperature=0,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    transcript = response.choices[0].message.content
    return str(transcript or "").strip(), latency_ms


def _local_template_values(
    audio_id: str,
    audio_path: pathlib.Path,
    prompt: str,
) -> dict[str, str]:
    return {
        "audio": str(audio_path),
        "audio_q": shlex.quote(str(audio_path)),
        "audio_abs": str(audio_path.resolve()),
        "audio_abs_q": shlex.quote(str(audio_path.resolve())),
        "audio_id": audio_id,
        "audio_stem": audio_path.stem,
        "audio_name": audio_path.name,
        "audio_parent": str(audio_path.parent),
        "prompt": prompt,
        "prompt_q": shlex.quote(prompt),
    }


def _render_local_template(
    template: Optional[str],
    values: dict[str, str],
    field_name: str,
) -> Optional[str]:
    if not template:
        return None
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder {{{exc.args[0]}}} in {field_name}"
        ) from exc


def transcribe_local_command(
    model_key: str,
    cfg: dict,
    audio_id: str,
    audio_path: pathlib.Path,
    prompt: str,
) -> tuple[str, float]:
    """
    Transcribe using a user-provided local command, e.g. a Parakeet wrapper.
    The command should print only the transcript to stdout unless output_file
    is configured.
    """
    values = _local_template_values(audio_id, audio_path, prompt)
    cwd_text = _render_local_template(cfg.get("cwd"), values, "cwd")
    cwd = pathlib.Path(cwd_text) if cwd_text else None
    command = _render_local_template(
        cfg.get("command_template"),
        values,
        "command",
    )
    output_file_text = _render_local_template(
        cfg.get("output_file_template"),
        values,
        "output_file",
    )

    t0 = time.monotonic()
    stdout = ""

    if command:
        argv = shlex.split(command, posix=True)
        if not argv:
            raise ValueError(f"Local command model {model_key!r} has an empty command")
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(cfg.get("timeout_seconds", DEFAULT_LOCAL_COMMAND_TIMEOUT_SECONDS)),
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"Local command failed with exit code {completed.returncode}: "
                f"{detail[-1000:]}"
            )
        stdout = completed.stdout.strip()

    latency_ms = (time.monotonic() - t0) * 1000

    if output_file_text:
        output_path = pathlib.Path(output_file_text)
        if cwd and not output_path.is_absolute():
            output_path = cwd / output_path
        transcript = output_path.read_text(encoding="utf-8").strip()
    else:
        transcript = stdout

    if not transcript:
        raise ValueError(
            f"Local command model {model_key!r} produced an empty transcript"
        )
    return transcript, latency_ms


def transcribe(
    model_key: str,
    audio_id: str,
    audio_path: pathlib.Path,
    variation: str = "",
    google_manual_mode: bool = False,
    openai_manual_mode: bool = False,
    local_manual_mode: bool = False,
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
            or (cfg["provider"] in {"local_openai_chat", "local_command"} and local_manual_mode)
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
            source = "api"
        elif cfg["provider"] == "openai":
            t, lat = transcribe_openai(cfg["model_id"], audio_path, prompt)
            source = "api"
        elif cfg["provider"] == "local_openai_chat":
            t, lat = transcribe_local_openai_chat(
                model_id=cfg["model_id"],
                audio_path=audio_path,
                prompt=prompt,
                base_url=cfg.get("base_url", DEFAULT_LOCAL_OPENAI_BASE_URL),
                api_key=cfg.get("api_key", DEFAULT_LOCAL_OPENAI_API_KEY),
                audio_mode=cfg.get("audio_mode", DEFAULT_LOCAL_AUDIO_MODE),
            )
            source = (
                "local_openai_chat:"
                f"{cfg.get('base_url', DEFAULT_LOCAL_OPENAI_BASE_URL)}"
            )
        elif cfg["provider"] == "local_command":
            t, lat = transcribe_local_command(
                model_key=model_key,
                cfg=cfg,
                audio_id=audio_id,
                audio_path=audio_path,
                prompt=prompt,
            )
            source = "local_command"
        else:
            raise ValueError(f"Unsupported transcription provider: {cfg['provider']}")
        return t, lat, None, source
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


# PER-SAMPLE EVALUATION

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
    """Call the evaluator model to evaluate a single (reference, hypothesis) pair."""
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
        resp = get_openai_client().chat.completions.create(
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


# GLOBAL SUMMARY

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


def collect_model_keys(
    all_results: list[dict],
    preferred_model_keys: Optional[list[str]] = None,
) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()

    for model_key in preferred_model_keys or []:
        if model_key not in seen:
            keys.append(model_key)
            seen.add(model_key)

    for result in all_results:
        for section in ("model_outputs", "evaluations"):
            for model_key in result.get(section, {}):
                if model_key not in seen:
                    keys.append(model_key)
                    seen.add(model_key)

    return keys


def generate_summary(
    all_results: list[dict],
    model_keys: Optional[list[str]] = None,
) -> dict:
    """Ask the evaluator model to produce the final benchmark summary."""
    model_keys = model_keys or collect_model_keys(
        all_results,
        list(TRANSCRIPTION_MODELS.keys()),
    )

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
                    for mk in model_keys
                },
            }
        )

    user_msg = (
        f"Here are transcription benchmark results for {len(all_results)} audio samples "
        f"(5 variations × 2 files) across {len(model_keys)} models.\n\n"
        f"Model keys: {model_keys}\n\n"
        f"Results:\n{json.dumps(condensed, indent=2)}\n\n"
        f"Produce a comprehensive summary following this exact JSON schema:\n{_SUMMARY_SCHEMA}"
    )

    try:
        resp = get_openai_client().chat.completions.create(
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


def compute_aggregates(
    all_results: list[dict],
    model_keys: Optional[list[str]] = None,
) -> dict:
    model_keys = model_keys or collect_model_keys(
        all_results,
        list(TRANSCRIPTION_MODELS.keys()),
    )
    stats: dict[str, dict] = {
        mk: {
            "wers": [],
            "cers": [],
            "mers": [],
            "latencies": [],
            "scores": [],
        }
        for mk in model_keys
    }

    variation_wers: dict[str, dict[str, list]] = {}

    for r in all_results:
        var = r["variation"]
        if var not in variation_wers:
            variation_wers[var] = {mk: [] for mk in model_keys}

        for mk in model_keys:
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


def load_existing_report(report_path: pathlib.Path) -> Optional[dict]:
    if not report_path.exists():
        return None
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError(f"Existing report has an invalid structure: {report_path}")
    return report


def merge_results(
    existing_results: list[dict],
    new_results: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    merged_results = [dict(result) for result in existing_results]
    index_by_audio_id: dict[str, int] = {}
    stats = {
        "audio_rows_added": 0,
        "audio_rows_updated": 0,
        "model_outputs_added": 0,
        "model_outputs_updated": 0,
        "evaluations_added": 0,
        "evaluations_updated": 0,
    }

    for idx, result in enumerate(merged_results):
        audio_id = str(result.get("audio_id", "")).strip()
        if audio_id and audio_id not in index_by_audio_id:
            index_by_audio_id[audio_id] = idx

    for new_result in new_results:
        audio_id = str(new_result.get("audio_id", "")).strip()
        if not audio_id or audio_id not in index_by_audio_id:
            merged_results.append(new_result)
            if audio_id:
                index_by_audio_id[audio_id] = len(merged_results) - 1
            stats["audio_rows_added"] += 1
            stats["model_outputs_added"] += len(new_result.get("model_outputs", {}))
            stats["evaluations_added"] += len(new_result.get("evaluations", {}))
            continue

        target = merged_results[index_by_audio_id[audio_id]]
        stats["audio_rows_updated"] += 1

        for key, value in new_result.items():
            if key not in {"model_outputs", "evaluations"}:
                target[key] = value

        target_outputs = target.setdefault("model_outputs", {})
        for model_key, output in new_result.get("model_outputs", {}).items():
            if model_key in target_outputs:
                stats["model_outputs_updated"] += 1
            else:
                stats["model_outputs_added"] += 1
            target_outputs[model_key] = output

        target_evaluations = target.setdefault("evaluations", {})
        for model_key, evaluation in new_result.get("evaluations", {}).items():
            if model_key in target_evaluations:
                stats["evaluations_updated"] += 1
            else:
                stats["evaluations_added"] += 1
            target_evaluations[model_key] = evaluation

    return merged_results, stats


def merge_transcription_model_metadata(existing_report: Optional[dict]) -> tuple[dict, dict]:
    existing_metadata = (existing_report or {}).get("benchmark_metadata", {})
    existing_models = existing_metadata.get("transcription_models", {})
    existing_details = existing_metadata.get("transcription_model_details", {})

    transcription_models: dict[str, str] = {}
    transcription_model_details: dict[str, dict] = {}
    if isinstance(existing_models, dict):
        transcription_models.update(existing_models)
    if isinstance(existing_details, dict):
        transcription_model_details.update(existing_details)

    transcription_models.update(
        {mk: cfg["display"] for mk, cfg in TRANSCRIPTION_MODELS.items()}
    )
    transcription_model_details.update(
        {
            mk: {
                key: value
                for key, value in cfg.items()
                if key not in {"api_key"}
            }
            for mk, cfg in TRANSCRIPTION_MODELS.items()
        }
    )
    return transcription_models, transcription_model_details


def build_report(
    all_results: list[dict],
    summary: dict,
    aggregates: dict,
    transcription_models: dict[str, str],
    transcription_model_details: dict[str, dict],
    google_manual_mode: bool,
    openai_manual_mode: bool,
    local_manual_mode: bool,
    manual_transcript_dir: pathlib.Path,
    append_report: bool,
    merge_stats: Optional[dict[str, int]] = None,
    previous_report_timestamp: Optional[str] = None,
) -> dict:
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluator_model": EVALUATOR_MODEL,
        "google_manual_mode": google_manual_mode,
        "openai_manual_mode": openai_manual_mode,
        "local_manual_mode": local_manual_mode,
        "manual_transcript_dir": str(manual_transcript_dir),
        "report_update_mode": "append" if append_report else "overwrite",
        "transcription_models": transcription_models,
        "transcription_model_details": transcription_model_details,
        "total_audio_samples": len(all_results),
        "dataset_variations": [
            "clean",
            "background_noise",
            "multiple_speakers",
            "accents",
            "multilingual",
        ],
        "samples_per_variation": 2,
        "metrics_used": [
            "WER",
            "CER",
            "MER",
            f"{EVALUATOR_MODEL} eval score (1-10)",
        ],
        "note_on_multilingual": (
            "WER is omitted for non-English samples; CER is reported instead. "
            f"{EVALUATOR_MODEL} evaluation judges quality holistically for all languages."
        ),
    }
    if merge_stats:
        metadata["append_merge_stats"] = merge_stats
    if previous_report_timestamp:
        metadata["previous_report_timestamp"] = previous_report_timestamp

    return {
        "benchmark_metadata": metadata,
        "aggregate_metrics": aggregates,
        "results": all_results,
        "summary": summary,
    }


def run_benchmark(
    google_manual_mode: bool = False,
    openai_manual_mode: bool = False,
    local_manual_mode: bool = False,
    manual_transcript_dir: pathlib.Path = DEFAULT_MANUAL_TRANSCRIPT_DIR,
    local_model_specs: Optional[list[str]] = None,
    local_command_model_specs: Optional[list[str]] = None,
    local_openai_base_url: str = DEFAULT_LOCAL_OPENAI_BASE_URL,
    local_openai_api_key: str = DEFAULT_LOCAL_OPENAI_API_KEY,
    local_audio_mode: str = DEFAULT_LOCAL_AUDIO_MODE,
    local_command_timeout_seconds: float = DEFAULT_LOCAL_COMMAND_TIMEOUT_SECONDS,
    model_filter: Optional[list[str]] = None,
    append_report: bool = False,
) -> None:
    if local_model_specs or local_command_model_specs:
        register_local_models(
            local_model_specs=local_model_specs,
            local_command_model_specs=local_command_model_specs,
            default_base_url=local_openai_base_url,
            default_api_key=local_openai_api_key,
            default_audio_mode=local_audio_mode,
            default_command_timeout_seconds=local_command_timeout_seconds,
        )

    if model_filter:
        missing_models = [key for key in model_filter if key not in TRANSCRIPTION_MODELS]
        if missing_models:
            raise ValueError(
                "Unknown model key(s) in --models: "
                + ", ".join(missing_models)
                + ". Available: "
                + ", ".join(TRANSCRIPTION_MODELS)
            )
        selected_models = {key: TRANSCRIPTION_MODELS[key] for key in model_filter}
        TRANSCRIPTION_MODELS.clear()
        TRANSCRIPTION_MODELS.update(selected_models)

    manual_transcript_dir = pathlib.Path(manual_transcript_dir)
    existing_report = load_existing_report(OUTPUT_PATH) if append_report else None
    manifest = load_manifest()
    logger.info(f"Loaded {len(manifest)} audio samples from manifest.json")
    if append_report:
        if existing_report:
            logger.info("Append report enabled. Merging into: %s", OUTPUT_PATH.resolve())
        else:
            logger.info(
                "Append report enabled, but no existing report was found at: %s",
                OUTPUT_PATH.resolve(),
            )
    if model_filter:
        logger.info("Model filter enabled: %s", ", ".join(TRANSCRIPTION_MODELS))
    if google_manual_mode:
        logger.info(
            "Google manual mode enabled. Reading manual transcripts from: %s",
            manual_transcript_dir.resolve(),
        )
    if local_model_specs or local_command_model_specs:
        local_keys = [
            key
            for key, cfg in TRANSCRIPTION_MODELS.items()
            if cfg["provider"] in {"local_openai_chat", "local_command"}
        ]
        logger.info(
            "Local models enabled (%s): %s",
            len(local_keys),
            ", ".join(local_keys),
        )
    if local_manual_mode:
        logger.info(
            "Local manual mode enabled. Reading manual transcripts from: %s",
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

        # Step 1: Transcribe with all configured models
        model_outputs: dict[str, dict] = {}

        for model_key in TRANSCRIPTION_MODELS:
            provider = TRANSCRIPTION_MODELS[model_key]["provider"]
            if (
                (
                    google_manual_mode
                    and provider == "google"
                )
                or (
                    openai_manual_mode
                    and provider == "openai"
                )
                or (
                    local_manual_mode
                    and provider in {"local_openai_chat", "local_command"}
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
                local_manual_mode=local_manual_mode,
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

    # Step 3: Merge with existing report if requested, then build global summary
    merge_stats = None
    previous_report_timestamp = None
    if append_report and existing_report:
        previous_report_timestamp = existing_report.get("benchmark_metadata", {}).get(
            "timestamp"
        )
        all_results, merge_stats = merge_results(
            existing_report.get("results", []),
            all_results,
        )
        logger.info(
            "Merged report rows: %s added, %s updated; model outputs: %s added, %s updated",
            merge_stats["audio_rows_added"],
            merge_stats["audio_rows_updated"],
            merge_stats["model_outputs_added"],
            merge_stats["model_outputs_updated"],
        )

    transcription_models, transcription_model_details = merge_transcription_model_metadata(
        existing_report if append_report else None
    )
    model_keys = collect_model_keys(all_results, list(transcription_models))
    for model_key in model_keys:
        transcription_models.setdefault(model_key, model_key)

    logger.info(f"\n{'─'*65}")
    logger.info(f"▶ Generating benchmark summary with {EVALUATOR_MODEL} ...")
    summary = generate_summary(all_results, model_keys=model_keys)
    aggregates = compute_aggregates(all_results, model_keys=model_keys)

    report = build_report(
        all_results=all_results,
        summary=summary,
        aggregates=aggregates,
        transcription_models=transcription_models,
        transcription_model_details=transcription_model_details,
        google_manual_mode=google_manual_mode,
        openai_manual_mode=openai_manual_mode,
        local_manual_mode=local_manual_mode,
        manual_transcript_dir=manual_transcript_dir,
        append_report=append_report,
        merge_stats=merge_stats,
        previous_report_timestamp=previous_report_timestamp,
    )

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
        display_name = transcription_models.get(mk, mk)
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
            f"    {display_name:<35} {wer_s:>9}  {score_s:>10}  {lat_s:>10}"
        )
    logger.info(f"{'═'*65}\n")


# ENTRY POINT

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        google_manual_mode=args.google_manual_mode,
        openai_manual_mode=args.openai_manual_mode,
        local_manual_mode=args.local_manual_mode,
        manual_transcript_dir=args.manual_transcript_dir,
        local_model_specs=args.local_model,
        local_command_model_specs=args.local_command_model,
        local_openai_base_url=args.local_openai_base_url,
        local_openai_api_key=args.local_openai_api_key,
        local_audio_mode=args.local_audio_mode,
        local_command_timeout_seconds=args.local_command_timeout_seconds,
        model_filter=_split_csv(args.models),
        append_report=args.append_report,
    )
