# 15: The custom batched scorer was barely validated

- **Folder:** `quant-noninferiority/`
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
The custom batched scorer was barely validated: 3 items per method, with log-prob drift up to 0.44 between batched and single. Padding effects could flip close calls; this is likely symmetric across arms but untested.

## Status
Open, documented. Nothing in the results was changed because of it. The README or writeup states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
