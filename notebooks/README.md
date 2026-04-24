# Reference notebook

`colab_original.py` is the **original Google Colab notebook export** this
project was derived from — a single-file pipeline that runs MediaTek
Breeze-ASR-25 on a Colab NVIDIA GPU via HuggingFace Transformers.

It is kept here as a reference for the "before" state. Do not try to run
it locally on Windows — it hard-codes `/content` paths, pulls
`google.colab.files`, and assumes CUDA. Use the `asr_local` CLI at the
repo root for local transcription.

**File status**: preserved as-is from the original Colab export. The
single-space indentation (a known Colab ipynb → .py export artefact)
means `python -m py_compile` on this file will fail — this is the
original upstream state and is intentional; the file is not part of
the test suite and not imported by any runtime code. A regex-based
emoji strip has been applied per the repo's no-emoji policy, but trace
emoji from outside the common Unicode ranges may remain; such traces
have no functional effect.
