# 16: Letter-logit zero-shot scoring is not representative of how Qwen3-8B is actually used

- **Folder:** `quant-noninferiority/`
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Letter-logit zero-shot scoring is not representative of how Qwen3-8B is actually used (chat/thinking).

## Status
Open, documented. Nothing in the results was changed because of it. The README or writeup states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
