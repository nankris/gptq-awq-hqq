# gptq-awq-hqq

Experiments 044 and 044b (GPTQ vs AWQ vs HQQ 4-bit quantization of Qwen3-8B). Experiment arms in this repo. Each folder has its own README, code, raw results, chart and verdict.

- [`044-gptq-awq-hqq`](044-gptq-awq-hqq/) Experiment 044: Three 4-bit recipes (GPTQ, AWQ, HQQ) on Qwen3-8B, 400 MMLU+ARC items — **INCONCLUSIVE**
- [`044b-quant-noninferiority`](044b-quant-noninferiority/) Experiment 044b: Can 4-bit Qwen3-8B stay within one accuracy point? (full MMLU + ARC-Challenge, 15,214 items) — **GPTQ INCONCLUSIVE, AWQ INCONCLUSIVE, HQQ DISPROVEN**
