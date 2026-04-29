"""
Kaggle script: transcribe 10 audio files with Gemma-4-E4B (audio-capable, ~4B params).

The Gemma-4 model code lives in transformers main but is NOT yet wired into
AutoConfig / AutoModel mappings, so we import the classes directly.

Setup (do this once when creating the notebook):
  - Accelerator: GPU T4 x2 (single T4 also works — fits in one 16GB card)
  - Internet: On
  - Environment: "Pin to original environment"  (gives torch 2.10+cu128)
  - Add Input: your audio benchmark dataset

Run order:
  1. Run CELL 1.
  2. Click the "Restart kernel" button at the top.  ← KERNEL RESTART, NOT
     "Factory reset" — factory reset wipes the pip install you just did.
  3. Run CELL 2.
"""

# ═════════════════════════════════════════════════════════════════════════════
# CELL 1 — install transformers from git, then KERNEL RESTART (not factory reset)
# ═════════════════════════════════════════════════════════════════════════════
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
     "--no-deps", "git+https://github.com/huggingface/transformers.git"],
    check=True,
)
# transformers HEAD pinned huggingface_hub >= 1.5.0; Kaggle ships 1.4.1.
# Bump only what's needed (not all deps, to avoid disturbing torch/numpy).
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", "-q",
     "huggingface_hub>=1.5.0"],
    check=True,
)
out = subprocess.run([sys.executable, "-m", "pip", "show", "transformers"],
                     capture_output=True, text=True).stdout
print(out)
print(">>> NOW: Run > 'Restart & clear cell outputs' (NOT Factory reset), then run CELL 2. <<<\n")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 2 — load model and transcribe (run AFTER kernel restart)
# ═════════════════════════════════════════════════════════════════════════════
import os
# Must be set BEFORE torch imports — reduces fragmentation between samples.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json, time, pathlib
import librosa, torch

# Verify the new transformers actually has Gemma-4. If this import fails,
# Cell 1 didn't take effect — re-run Cell 1 and kernel-restart again.
from transformers.models.gemma4 import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4Processor,
)

# ── config ───────────────────────────────────────────────────────────────────
HF_TOKEN     = "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"
MODEL_ID     = "google/gemma-4-E4B-it"
INPUT_ROOT   = pathlib.Path("/kaggle/input")    # searched recursively
STATE_FILE   = pathlib.Path("/kaggle/working/gemma4_results_partial.json")
RESET        = False   # set to True to ignore STATE_FILE and re-run everything
REPO_ROOT    = pathlib.Path("/kaggle/working/llm-audio-transcription-benchmark")
MANIFEST_PATH = REPO_ROOT / "manifest.json"

MANIFEST = [
    {"id": "clean_1",         "file": "6829-68769-0000.wav",  "variation": "clean"},
    {"id": "clean_2",         "file": "5639-40744-0031.wav",  "variation": "clean"},
    {"id": "noise_1",         "file": "83-11691-0035.wav",    "variation": "background_noise"},
    {"id": "noise_2",         "file": "196-122150-0032.wav",  "variation": "background_noise"},
    {"id": "multi_speaker_1", "file": "EN2002a.wav",          "variation": "multiple_speakers"},
    {"id": "multi_speaker_2", "file": "TS3003d.wav",          "variation": "multiple_speakers"},
    {"id": "accent_1",        "file": "hindko1.mp3",          "variation": "accents"},
    {"id": "accent_2",        "file": "telugu1.mp3",          "variation": "accents"},
    {"id": "multilingual_1",  "file": "multilingual_1.wav",   "variation": "multilingual"},
    {"id": "multilingual_2",  "file": "multilingual_2.wav",   "variation": "multilingual"},
]

SINGLE_PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write digits (e.g. write 3 instead of three)."
)
MULTI_PROMPT = (
    "Transcribe this multi-speaker audio in the original language. "
    "Preserve speaker turns using labels SPEAKER_1:, SPEAKER_2:, etc., "
    "one turn per line. Output only the labeled transcript — no timestamps "
    "or explanations."
)

def find_audio(filename: str):
    matches = list(INPUT_ROOT.rglob(filename))
    return matches[0] if matches else None


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest.json not found at {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {entry["id"]: entry for entry in manifest}


def compute_metrics(reference: str, hypothesis: str, language: str):
    try:
        import jiwer
    except Exception:
        import sys, subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "jiwer"],
            check=True,
        )
        import jiwer

    if not reference or not hypothesis:
        return (
            1.0 if language == "en" else None,
            1.0 if hypothesis == "" else None,
            1.0 if language == "en" else None,
        )

    cer = jiwer.cer(reference, hypothesis)
    if language != "en":
        return (None, cer, None)
    return (jiwer.wer(reference, hypothesis), cer, jiwer.mer(reference, hypothesis))

# ── load model directly (auto-mapping doesn't know about gemma4 yet) ─────────
print(f"transformers={__import__('transformers').__version__}  "
      f"torch={torch.__version__}  cuda={torch.version.cuda}  "
      f"devices={torch.cuda.device_count()}")
print(f"Loading {MODEL_ID}...")

# --- BEFORE LOADING MODEL ---
import torch, gc
gc.collect()
torch.cuda.empty_cache()

# --- LOAD MODEL ---
processor = Gemma4Processor.from_pretrained(MODEL_ID, token=HF_TOKEN)

model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",                 # more stable than balanced here
    max_memory={0: "11GiB", 1: "11GiB", "cpu": "20GiB"},  # leave headroom
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
).eval()

print("Model loaded.\n")
# ── transcribe (resumable: skip entries already in STATE_FILE) ──────────────
if STATE_FILE.exists() and not RESET:
    saved = json.loads(STATE_FILE.read_text())
    done_by_id = {r["audio_id"]: r for r in saved
                  if r.get("transcript") and not r.get("error")}
    print(f"Resuming: {len(done_by_id)}/{len(MANIFEST)} already transcribed.\n")
else:
    done_by_id = {}
    if RESET and STATE_FILE.exists():
        STATE_FILE.unlink()

manifest_by_id = {entry["id"]: entry for entry in MANIFEST}
results = []
for entry in MANIFEST:
    if entry["id"] in done_by_id:
        results.append(done_by_id[entry["id"]])
        print(f"[{entry['id']}] cached, skipping")
        continue

    path = find_audio(entry["file"])
    if path is None:
        print(f"[SKIP] {entry['id']} — {entry['file']} not found under /kaggle/input")
        results.append({"audio_id": entry["id"], "transcript": "",
                        "latency_ms": 0, "error": "file not found"})
        continue

    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    prompt = MULTI_PROMPT if entry["variation"] == "multiple_speakers" else SINGLE_PROMPT

    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text",  "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    print(f"[{entry['id']}] transcribing...", end=" ", flush=True)
    t0 = time.monotonic()
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        if hasattr(processor, "parse_response"):
            raw = processor.decode(out[0][input_len:], skip_special_tokens=False)
            parsed = processor.parse_response(raw)
            # parse_response can return a dict, list of dicts, or a string
            if isinstance(parsed, dict):
                text = parsed.get("text") or parsed.get("content") or str(parsed)
            elif isinstance(parsed, list):
                text = " ".join(
                    (p.get("text") or p.get("content") or "")
                    if isinstance(p, dict) else str(p)
                    for p in parsed
                )
            else:
                text = str(parsed)
        else:
            text = processor.decode(out[0][input_len:], skip_special_tokens=True)
        text = (text or "").strip()
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        manifest_entry = manifest_by_id.get(entry["id"], {})
        reference = manifest_entry.get("reference", "")
        language = manifest_entry.get("language", "en")
        wer, cer, mer = compute_metrics(reference, text, language)
        results.append({
            "audio_id": entry["id"],
            "transcript": text,
            "latency_ms": latency_ms,
            "error": None,
            "wer": wer,
            "cer": cer,
            "mer": mer,
            "transcription_source": "kaggle_gemma4",
        })
        print(f"OK ({latency_ms:.0f}ms)")
    except Exception as exc:
        manifest_entry = manifest_by_id.get(entry["id"], {})
        reference = manifest_entry.get("reference", "")
        language = manifest_entry.get("language", "en")
        wer, cer, mer = compute_metrics(reference, "", language)
        results.append({
            "audio_id": entry["id"],
            "transcript": "",
            "latency_ms": 0,
            "error": str(exc),
            "wer": wer,
            "cer": cer,
            "mer": mer,
            "transcription_source": "kaggle_gemma4",
        })
        print(f"ERROR: {exc}")

    # Free the per-sample tensors and KV cache before the next iteration —
    # otherwise they accumulate and OOM the second sample.
    try:
        del inputs, out
    except NameError:
        pass
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Persist progress so KeyboardInterrupt / OOM doesn't lose finished work.
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))

# ── output ───────────────────────────────────────────────────────────────────
print("\n\n=== PASTE THIS BLOCK BACK INTO CLAUDE ===\n")
print(json.dumps({"model_key": "gemma-4-local", "results": results},
                 ensure_ascii=False, indent=2))
