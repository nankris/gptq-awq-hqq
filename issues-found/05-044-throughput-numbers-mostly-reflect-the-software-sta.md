# 05 (044): Throughput numbers mostly reflect the software stack

- **Experiment:** 044 (`044-gptq-awq-hqq/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `044-gptq-awq-hqq.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Throughput numbers mostly reflect the software stack**, not the methods: AWQ was decompressed, HQQ ran at 2.4 tok/s on a PyTorch path, GPTQ used TorchLinear. The writeup flags this clearly. The resources.png figure still invites misreading.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
