# VibeThinker-3B — Hands-On Notebook

A practical, runnable Jupyter notebook for getting started with [`WeiboAI/VibeThinker-3B`](https://huggingface.co/WeiboAI/VibeThinker-3B) — a 3B-parameter reasoning model that punches far above its weight on **verifiable** tasks like competition math, coding, and STEM.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalayogiG/vibethinker-3b-hands-on/blob/main/Hands_on_VibeThinker_3B.ipynb)
[![Model](https://img.shields.io/badge/🤗%20Model-VibeThinker--3B-yellow)](https://huggingface.co/WeiboAI/VibeThinker-3B)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## What is VibeThinker-3B?

VibeThinker-3B is a 3B-parameter dense reasoning model from WeiboAI, fine-tuned from `Qwen2.5-3B`. Despite its small size, it reaches the accuracy range of much larger frontier reasoning models on **verifiable** benchmarks — AIME, HMMT, IMO-AnswerBench, LiveCodeBench, and recent LeetCode contests.

The model is built around the **Parametric Compression-Coverage Hypothesis**: multi-step reasoning with reliable feedback signals is a *compressible*, parameter-dense capability, so a compact model can carry near-frontier reasoning ability. Broad open-domain world knowledge still benefits from scale — so this is a specialist, not a general-purpose chatbot.

| Property | Value |
|----------|-------|
| Parameters | 3B (dense) |
| Base model | Qwen2.5-3B |
| Precision | BF16 (~6 GB weights) |
| Context window | 64K (trained) |
| Max generation | up to ~102K tokens |
| License | MIT |
| Best at | math, coding, STEM — verifiable reasoning |
| Not built for | open-domain knowledge, general chat, multimodal, tool-use orchestration |

---

## What's in the notebook

The notebook (`Hands_on_VibeThinker-3B.ipynb`) walks through, end to end:

1. **Environment setup** — requirements and version checks (`transformers >= 4.54.0`)
2. **Loading** the model and tokenizer in BF16
3. **A reusable `VibeThinker` wrapper** matching the official inference recipe
4. **Worked examples** — competition math, coding, and STEM reasoning
5. **Streaming generation** with `TextIteratorStreamer`
6. **Inspecting the reasoning trace** (splitting `<think>` blocks from the final answer)
7. **Batched inference** with correct left-padding
8. **Practical tips & gotchas**, including notes on thinking-token control

---

## Quick start

### Run on Google Colab (easiest)
Click the **Open in Colab** badge above. Select a GPU runtime (`Runtime → Change runtime type → GPU`) and run the cells top to bottom.

### Run locally
```bash
git clone https://github.com/BalayogiG/vibethinker-3b-hands-on.git
cd vibethinker-3b-hands-on

pip install "transformers>=4.54.0" accelerate torch jupyter

jupyter notebook Hands_on_VibeThinker-3B.ipynb
```

### Minimal inference snippet
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_path = "WeiboAI/VibeThinker-3B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="bfloat16", device_map="auto", low_cpu_mem_usage=True
)

messages = [{"role": "user", "content": "Prove there are infinitely many primes."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

out = model.generate(**inputs, generation_config=GenerationConfig(
    max_new_tokens=8192, do_sample=True, temperature=1.0, top_p=0.95, top_k=None,
))
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## Recommended sampling settings

The technical report evaluates with these — **use them for best results**:

| Param | Value |
|-------|-------|
| `temperature` | `1.0` |
| `top_p` | `0.95` |
| `top_k` | `-1` / `None` (disabled) |
| `max_new_tokens` | up to ~100K for hard problems |

> ⚠️ **Counter-intuitive:** lowering the temperature toward 0 tends to *hurt* this model on reasoning tasks. It is tuned for `temperature=1.0`.

---

## A note on "turning off thinking"

VibeThinker-3B is **not** a hybrid model — there's no `enable_thinking=False` switch. Reasoning is intrinsic to the model. You can *bound* the thinking with `max_new_tokens` or prompt for brevity, but you can't cleanly disable it. The authors instead trained reasoning efficiency into the model via **Long2Short RL** (rewarding shorter correct responses). If you need fast, short answers for easy turns, route those to a general-purpose model and reserve VibeThinker for verifiable reasoning.

---

## Hardware

- BF16 weights ≈ 6 GB. A GPU with **≥ 12 GB VRAM** is comfortable.
- Reasoning traces can be very long; budget extra memory for the KV cache, or use **vLLM** for efficient long-context inference.
- CPU works but is slow for long generations.

---

## Resources

- 🤗 Model card: https://huggingface.co/WeiboAI/VibeThinker-3B
- 💻 Official repo: https://github.com/WeiboAI/VibeThinker
- 📄 Technical report: https://arxiv.org/abs/2606.16140

---

## License

This repository's notebook and docs are released under the **MIT License** (matching the model's license). The VibeThinker-3B model weights are © WeiboAI, also MIT-licensed — see the model card for details.

## Citation

```bibtex
@misc{xu2026vibethinker3b,
  title={VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models},
  author={Sen Xu and Shixi Liu and Wei Wang and Jixin Min and Yingwei Dai and
          Zhibin Yin and Yirong Chen and Xin Zhou and Junlin Zhang},
  year={2026},
  eprint={2606.16140},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2606.16140}
}
```

---

*This is an unofficial community notebook and is not affiliated with WeiboAI.*
