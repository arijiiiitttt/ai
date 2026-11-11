# my-slm

Train a small language model whenever you want, keep it on Hugging Face, use it anywhere.

## Setup (once)
    pip install numpy tokenizers safetensors huggingface_hub jax optax
    export HF_TOKEN=your_write_token      # huggingface.co/settings/tokens  (Windows: set HF_TOKEN=...)

## Train / resume (pushes to Hugging Face automatically)
    python slm.py train --repo YOURNAME/my-slm --data mytext.txt --steps 2000
Run it again any time with more data to continue training the same model. Add --fresh to start over.

## Use
    python slm.py pull --repo YOURNAME/my-slm
    python slm.py gen  --repo YOURNAME/my-slm "Once upon a time"

## Online demo (free Hugging Face Space)
    python slm.py space --repo YOURNAME/my-slm --space YOURNAME/my-slm-demo

## Other files
- model_int8.npz + tokenizer.json + run.py: the first 3 MB demo model trained in the chat
  (python run.py "text" --tokens 100). It is a toy; it does not know facts.
