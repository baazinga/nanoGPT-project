# nanoGPT from Scratch and KV Cache Performance Study

An educational, character-level GPT project developed while following Andrej
Karpathy's *Let's build GPT: from scratch, in code, spelled out*. The project
has two parts:

1. a small decoder-only Transformer trained on the Tiny Shakespeare corpus;
2. a later course study of how KV caching affects autoregressive inference time
   and peak GPU memory under different sequence lengths and batch sizes.

The repository keeps the implementation, presentation, and experiment figures
together while distinguishing what is currently reproducible from what is
preserved as a course-work record.

## Project status

| Component | Status |
| --- | --- |
| Character-level decoder-only Transformer | Source code included |
| Training and text generation | Source code and dataset included |
| KV-cache performance analysis | Presentation and figures included |
| KV-cache benchmark implementation and raw logs | Not currently included |

The supplied local `KVcache.py` was byte-identical to the baseline `train.py`
and did not contain cache-aware inference. It has therefore not been presented
as a KV-cache implementation. The numerical results in the presentation should
be treated as documented course-project results rather than a fully
reproducible benchmark until the original server-side code and logs are added.

## Baseline model

The baseline implements the main components of a compact GPT-style language
model in PyTorch:

- character-level tokenization;
- token and positional embeddings;
- masked single-head and multi-head self-attention;
- feed-forward layers, residual connections, layer normalization, and dropout;
- next-token training and autoregressive text generation.

This is an educational implementation built step by step from Karpathy's
lecture, not a claim of an independently invented architecture.

## KV-cache study

The follow-up study asks two questions:

1. Under what sequence-length and batch-size settings does KV caching provide
   the clearest inference-time improvement?
2. How does caching affect peak memory during priming and token generation?

The course presentation reports experiments across multiple sequence lengths
and batch sizes, with repeated runs for each configuration. It records that
larger batches and longer contexts benefited most in the tested setting, while
small-batch overhead could offset the benefit. It also separates the memory
cost of storing keys and values from the activation cost avoided during
generation.

The original presentation is preserved unchanged at
[`docs/KV_Cache_Performance_Analysis_2025-12-12.pptx`](docs/KV_Cache_Performance_Analysis_2025-12-12.pptx).
Its title slide records the student name, class, and presentation date of
12 December 2025. A historical repository URL shown in the slides was a planned
project location and is not the current repository URL.

## Repository layout

```text
.
├── train.py                         # baseline Transformer training script
├── input.txt                        # Tiny Shakespeare training text
├── requirements.txt
├── docs/
│   ├── KV_Cache_Performance_Analysis_2025-12-12.pptx
│   └── figures/                     # plots used in the performance analysis
└── nanoGPT-project/                 # files retained from the original upload
```

The `nanoGPT-project/` directory and the earlier report files are retained to
preserve the repository's original history. The root-level files provide the
clean entry point for the reorganized version.

## Running the baseline

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

The script trains from scratch and then samples generated text. Its default
configuration is deliberately small and can run on CPU, although training may
take time. The script expects `input.txt` in the repository root.

## Reproducibility notes

- The random seed is fixed in `train.py`.
- The baseline script is preserved close to the submitted course-project
  version instead of being retrospectively rewritten.
- Exact KV-cache numbers require the original benchmark code, hardware setup,
  and raw logs, which are not currently available in this repository.
- No pretrained weights or checkpoints are included.

## Timeline

- **June 2024:** baseline GPT implementation, diagrams, and intermediate
  screenshots recorded in the original upload.
- **12 December 2025:** KV-cache performance-analysis presentation completed.

## Attribution

The baseline was developed as a learning exercise alongside:

- Andrej Karpathy, [*Let's build GPT: from scratch, in code, spelled
  out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Andrej Karpathy, [`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT)

Additional references used for the KV-cache study are listed in the
presentation.

## Author

Jiayin Tian, Xi'an Jiaotong University

