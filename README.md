# gptq-awq-hqq

GPTQ vs AWQ vs HQQ 4-bit quantization of Qwen3-8B, in two independent studies. Each folder has its own README, code, raw results, charts and verdict.

- [`original-run`](original-run/): Three 4-bit recipes (GPTQ, AWQ, HQQ) on Qwen3-8B, 400 MMLU+ARC items. **INCONCLUSIVE**
- [`quant-noninferiority`](quant-noninferiority/): Can 4-bit Qwen3-8B stay within one accuracy point? Full MMLU + ARC-Challenge, 15,214 items. **GPTQ INCONCLUSIVE, AWQ INCONCLUSIVE, HQQ DISPROVEN**
- [`issues-found`](issues-found/): known limitations of both studies, one file per issue.
