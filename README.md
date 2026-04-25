# 🎙️ Multi-Model ASR Transcription Benchmark

Benchmarks **Gemma-4**, **Gemini 3.1 Pro Preview**, **Gemini 3 Flash Preview**, and **GPT-4o Transcribe**
across 5 audio variations (10 files total), evaluated by **GPT-5.5**.

---

## 📦 Setup

```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
transcription_benchmark/
├── benchmark.py          ← Main script
├── manifest.json         ← Audio file index + reference transcripts
├── requirements.txt
├── audio_files/          ← Audio files you download (10 total, 2 per variation)
│   ├── clean_1.flac
│   ├── clean_2.flac
│   ├── noise_1.flac
│   ├── noise_2.flac
│   ├── multi_speaker_1.wav
│   ├── multi_speaker_2.wav
│   ├── accent_1.wav
│   ├── accent_2.wav
│   ├── multilingual_1.wav  (Hindi)
│   └── multilingual_2.wav  (French)
└── benchmark_report.json ← Generated output
```

---

## 🌐 Audio Dataset Download Sources

### 1 — Clean Baseline: LibriSpeech `test-clean`

|                     |                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------- |
| **URL**             | https://www.openslr.org/12/                                                            |
| **Direct download** | `wget https://www.openslr.org/resources/12/test-clean.tar.gz`                          |
| **Size**            | ~346 MB                                                                                |
| **Format**          | FLAC, 16 kHz, mono                                                                     |
| **Transcripts**     | Each chapter folder contains a `.trans.txt` file mapping `<utterance-id> <transcript>` |

**Steps:**

1. Download and extract `test-clean.tar.gz`
2. Pick any 2 `.flac` files (e.g. from `LibriSpeech/test-clean/1089/134686/`)
3. Copy their transcript lines from the `.trans.txt` file into `manifest.json`

---

### 2 — Background Noise: LibriSpeech `test-other`

|                      |                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------- |
| **URL**              | https://www.openslr.org/12/                                                              |
| **Direct download**  | `wget https://www.openslr.org/resources/12/test-other.tar.gz`                            |
| **Size**             | ~328 MB                                                                                  |
| **Format**           | FLAC, 16 kHz, mono                                                                       |
| **Why `test-other`** | Acoustically harder recordings (non-standard accents, room noise, mic quality variation) |

> **Alternative for real CHiME noise:** CHiME-6 requires free registration at https://chimechallenge.org/
> Download the `chime6_audio` split and use the JSON annotations for ground-truth transcripts.

---

### 3 — Multiple Speakers: AMI Corpus (Headset Mix)

|                       |                                                     |
| --------------------- | --------------------------------------------------- |
| **URL**               | https://groups.inf.ed.ac.uk/ami/download/           |
| **Downloader script** | https://github.com/BUTSpeechFIT/AMICorpusDownloader |
| **Format**            | WAV, 16 kHz                                         |
| **Transcripts**       | Word-level `.words.xml` files per meeting           |

**Steps:**

1. Use the downloader: `python download_ami.py --subset IHM --meetings EN2001a EN2001b`
2. Pick 2 meeting segments (30–90 seconds) from `IHM` (individual headset mix)
3. Extract the transcript by concatenating word elements from the `.words.xml` annotation

> **Easier alternative:** Use HuggingFace:
>
> ```python
> from datasets import load_dataset
> ds = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
> ```
>
> Each example has `audio` + `text` fields.

---

### 4 — Accents: L2-ARCTIC Corpus

|                     |                                                                              |
| ------------------- | ---------------------------------------------------------------------------- |
| **URL**             | https://psi.engr.tamu.edu/l2-arctic-corpus/                                  |
| **Direct download** | Fill the form on the page → receive download link via email                  |
| **Format**          | WAV, 44.1 kHz (downsample to 16 kHz recommended)                             |
| **Speakers**        | 24 non-native speakers: Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese |
| **Transcripts**     | Prompt files — every utterance maps to a line in `prompts/` folder           |

> **Faster alternative — Speech Accent Archive:**
> https://accent.gmu.edu/ — Browse by language, download individual MP3 recordings.
> Reference text (same passage for all speakers): _"Please call Stella..."_ paragraph.

> **HuggingFace alternative:**
>
> ```python
> from datasets import load_dataset
> ds = load_dataset("speech_accent_archive", split="train")
> ```

---

### 5 — Multilingual: FLEURS (Google)

|                    |                                               |
| ------------------ | --------------------------------------------- |
| **URL**            | https://huggingface.co/datasets/google/fleurs |
| **Languages used** | `hi_in` (Hindi), `fr_fr` (French)             |

**Download via HuggingFace:**

```python
from datasets import load_dataset
import soundfile as sf
import numpy as np

for lang, out_name in [("hi_in", "multilingual_1"), ("fr_fr", "multilingual_2")]:
    ds = load_dataset("google/fleurs", lang, split="test", streaming=True)
    sample = next(iter(ds))

    audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
    sr = sample["audio"]["sampling_rate"]
    sf.write(f"audio_files/{out_name}.wav", audio_array, sr)

    print(f"{out_name}: {sample['transcription']}")
    # → paste this into manifest.json as the reference
```

---

## ✏️ Updating manifest.json

After downloading files, update the `"reference"` field for each entry in `manifest.json`:

```json
{
    "id": "clean_1",
    "variation": "clean",
    "file": "audio_files/clean_1.flac",
    "reference": "he hoped there would be stew for dinner turnips and carrots and bruised potatoes"
}
```

---

## 🚀 Running the Benchmark

```bash
python benchmark.py
```

This will:

1. Transcribe each of the 10 audio files with all 4 models
2. Compute WER, CER, MER per model per file
3. Call GPT-5.5 to evaluate each (reference, hypothesis) pair
4. Generate a comprehensive GPT-5.5 summary
5. Write `benchmark_report.json`

---

## 📄 Output Schema: `benchmark_report.json`

```
{
  "benchmark_metadata": { ... },
  "aggregate_metrics": {
    "<model_key>": {
      "avg_wer": 0.0,
      "avg_cer": 0.0,
      "avg_latency_ms": 0.0,
      "avg_eval_score": 0.0,
      "per_variation_avg_wer": { "clean": 0.0, "background_noise": 0.0, ... }
    }
  },
  "results": [
    {
      "audio_id": "clean_1",
      "variation": "clean",
      "audio_file": "audio_files/clean_1.flac",
      "audio_mime_type": "audio/flac",
      "audio_base64": "<base64 for UI playback>",
      "reference_transcript": "...",
      "model_outputs": {
        "gemma-4":                { "transcript": "...", "wer": 0.0, "cer": 0.0, "latency_ms": 0 },
        "gemini-3.1-pro-preview": { ... },
        "gemini-3-flash-preview": { ... },
        "gpt-4o-transcribe":      { ... }
      },
      "evaluations": {
        "gemma-4": {
          "accuracy_score": 8,
          "fluency_score": 9,
          "completeness_score": 8,
          "overall_score": 8,
          "failure_summary": "...",
          "error_categories": ["substitution_errors"],
          "improvement_suggestions": "...",
          "notable_errors": ["..."]
        },
        ...
      }
    }
  ],
  "summary": {
    "overall_ranking": [ { "rank": 1, "model": "...", "verdict": "..." } ],
    "per_variation_winner": { "clean": "...", ... },
    "model_analysis": {
      "<model_key>": {
        "strengths": [...],
        "weaknesses": [...],
        "failure_patterns": "...",
        "how_to_improve": "...",
        "overall_verdict": "..."
      }
    },
    "benchmark_insights": [...],
    "recommendations": [...]
  }
}
```

The `audio_base64` field lets your UI render an `<audio>` element:

```html
<audio controls src="data:audio/flac;base64,{audio_base64}" />
```

---

## 📊 Metrics Reference

| Metric            | Meaning                                                  | Lower = better       |
| ----------------- | -------------------------------------------------------- | -------------------- |
| **WER**           | Word Error Rate — (S+D+I)/N                              | ✅                   |
| **CER**           | Character Error Rate — character-level WER               | ✅                   |
| **MER**           | Match Error Rate — accounts for multiple alignment paths | ✅                   |
| **GPT-5.5 score** | Holistic quality score 1–10                              | ❌ (higher = better) |

> WER/MER are only computed for English samples.
> CER is computed for all languages including multilingual samples.
