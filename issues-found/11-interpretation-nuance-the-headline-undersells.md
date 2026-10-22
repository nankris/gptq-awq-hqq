# 11: Interpretation nuance the headline undersells

- **Folder:** `quant-noninferiority/`
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Interpretation nuance the headline undersells**: GPTQ and AWQ CIs exclude zero (UB -0.45 / -0.55), so there is a *statistically clear* accuracy loss, just not clearly beyond 1 pp. The writeup does say "An interval can exclude zero while crossing -1". The point estimates (-0.90, -0.97) sit right at the margin, so even 10x more data would probably stay INCONCLUSIVE. The 1 pp margin is arbitrary, and the verdict depends entirely on it.

## Status
Open, documented. Nothing in the results was changed because of it. The README or writeup states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
