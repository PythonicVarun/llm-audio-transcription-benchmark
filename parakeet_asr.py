#!/usr/bin/env python3
import logging
import os
import sys

# Suppress NeMo and third-party noise before any imports
logging.disable(logging.CRITICAL)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import nemo.collections.asr as nemo_asr  # noqa: E402

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: parakeet_asr.py <audio_file>")

    audio_path = sys.argv[1]
    # Load on CPU first so float32 weights don't fill the 4 GB VRAM,
    # then convert to float16 (~1.8 GB) before moving to GPU.
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu")
    model = model.half().cuda()
    model.eval()

    transcripts = model.transcribe([audio_path])
    # transcribe() returns a list; entries may be strings or hypothesis objects
    result = transcripts[0]
    text = result.text if hasattr(result, "text") else str(result)
    print(text.strip())
