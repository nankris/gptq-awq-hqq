# 04 (044): Timing comparison is unequal

- **Experiment:** 044 (`044-gptq-awq-hqq/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `044-gptq-awq-hqq.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Timing comparison is unequal**: GPTQ time includes first-use CUDA extension compilation and uses layer offload (2.8 GiB peak). AWQ is GPU-resident (21 GiB). HQQ loads the full BF16 model onto the GPU (16.2 GiB). "HQQ fastest" is real by a wide margin (41 s vs 315 s vs 840 s), so the ranking probably holds.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
