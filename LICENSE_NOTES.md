# License notes: TabPFN-3 (Part B item 4)

This document is specific to the TabPFN-3 migration and is separate from
any top-level repository LICENSE file, because the code in this
repository, the `tabpfn` Python package's code, and the TabPFN-3 model
weights are covered by three different licenses.

## Three separate licenses are in play

1. **This repository's own code** (`src/mna_pipeline`, tests, etc.):
   whatever top-level LICENSE this repository declares. Unaffected by
   this migration.

2. **The `tabpfn` Python package's source code** (installed via
   `pip install tabpfn==8.1.0`): the "Prior Labs License (Apache 2.0
   with ADDITIONAL PROVISION)", version 1.2, Dec 2025. Confirmed by
   reading `pip show tabpfn`'s License field directly (checked
   2026-07-15). This is Apache 2.0 plus one additional attribution
   requirement (Section 10): if you distribute or make available the
   package, its source, or a product built on it, you must include a
   copy of the license and display "Built with PriorLabs-TabPFN" in
   related UI/docs, and any AI model trained using the source/weights
   must start its name with "TabPFN" if that model is itself
   distributed or made available. Section 10 explicitly exempts
   "internal benchmarking and testing without external communication."

3. **The TabPFN-3 model checkpoint weights themselves**
   (`tabpfn-v3-classifier-v3_20260417_binary.ckpt`, downloaded from the
   gated HuggingFace repo `Prior-Labs/tabpfn_3`): licensed under the
   **"TABPFN-3 License v1.0"**, which is a separate, non-commercial
   license from the code license above. Confirmed via Prior Labs'
   documentation (https://docs.priorlabs.ai/models, checked
   2026-07-15) and corroborated by web search results describing the
   license as "permissive for research and internal evaluation" while
   prohibiting "commercial or production purpose, including
   revenue-generating products, competitive benchmarking for
   procurement, client deliverables, or using the model's results for
   internal commercial decision-making." Downloading the weights
   requires interactive license acceptance at https://ux.priorlabs.ai
   (or a `TABPFN_TOKEN` for non-interactive environments); this
   repository's automated sandbox has no such token and could not
   complete a real weight download, so no weights were embedded in or
   distributed with this repository (see README "Known Limitations").

   This repository's own attempt to fetch and directly quote the full
   TABPFN-3 License v1.0 legal text did not succeed (huggingface.co was
   not reachable from this environment, and no canonical license-text
   URL was found on priorlabs.ai). The characterization above is based
   on Prior Labs' documentation and search-indexed summaries, not a
   direct reading of the full legal text. **Before using this
   pipeline's TabPFN-3 results for anything beyond academic coursework,
   independently confirm the exact current TABPFN-3 License v1.0 text
   at https://ux.priorlabs.ai or by contacting Prior Labs directly.**

## Is this thesis/academic use compliant?

Based on the above: yes, with the caveat in the paragraph immediately
above. A master's thesis at University College Dublin is:

- **Research/academic use**: explicitly within the described permitted
  scope ("research and internal evaluation").
- **Not commercial**: no revenue-generating product, no client
  deliverable, no procurement benchmarking, no commercial decision
  driven by the model's output.
- **Not redistributing the weights**: the thesis reports metrics
  produced by the model; it does not ship the `.ckpt` file itself.

If any part of this work is later reused in a context that could be
read as commercial (e.g., a startup built on these findings, or a paid
consulting deliverable), the TABPFN-3 License v1.0 terms above indicate
that would require Prior Labs' separate Commercial Enterprise License,
and this should be re-checked directly with Prior Labs before that
reuse.

## Attribution

Per the code license's Section 10 exemption, purely internal academic
benchmarking does not require the "Built with PriorLabs-TabPFN"
attribution. This README nonetheless credits Prior Labs and TabPFN-3
explicitly (see README.md "Model identity") as a matter of academic
citation practice, independent of the license's legal attribution
trigger.
