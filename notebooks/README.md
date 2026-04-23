# Reference notebook

`colab_original.py` is the **original Google Colab notebook export** this
project was derived from — a single-file pipeline that runs MediaTek
Breeze-ASR-25 on a Colab NVIDIA GPU via HuggingFace Transformers.

It is kept here as a reference for the "before" state. Do not try to run
it locally on Windows — it hard-codes `/content` paths, pulls
`google.colab.files`, and assumes CUDA. Use the `asr_local` CLI at the
repo root for local transcription.
