# Audit report — what changed in airtight-v2 → arXiv v1

Background: six independent reviewer agents audited `paper-airtight-v2.pdf` for statistical claims, hedging, math, citations, writing, and reproducibility. This document records what was fixed.

## Critical fixes

| # | Issue | Resolution |
|---|---|---|
| 1 | Body typo "as a opponent model" (Future Work bullet) | `as an opponent model` |
| 2 | Figure 1 bar chart only had "No OM" x-axis label | Explicit `xtick={...}` listing all 5 categorical positions |
| 3 | Figure 3 caption smuggled a §4.5 result into the architecture diagram | Caption now describes the diagram only; result moved out, reference to §4.5 retained |
| 4 | §6.3 retracted the "Oracle Paradox" framing while §4.8 #3 still called it "fundamental consequence" — direct self-contradiction | Kept the label (it's the figure-caption anchor) but softened: §6.3 now says we retain the term as a memorable handle, §4.8 #3 says "robust property of the cooperative blending formula" |
| 5 | "Oguntola et al. 2023 ToM2C" citation — arXiv:2111.09189 is actually Wang et al. 2022 (ICLR) | Fixed to Wang et al. 2022 with correct ICLR venue |
| 6 | Packet wording "all 38 per-game JSON logs" — actually 36 JSONs + 2 checkpoints = 38 manifest entries | Reworded; DOI/Zenodo plan added |

## Statistical / numerical fixes

- **Holm-Bonferroni "10 comparisons" never enumerated.** Added Table 13 (`tab:holm`) listing all 10 pairs with Δ, p_Holm, sig flags. Ran `stats_tests.py` on packet logs to get exact values — 8 of 10 sig at p=0.001, 2 NS as expected.
- **SE 0.4pp claim understated for σ=8.9% (Cond A) comparisons.** Now stated as "~0.4pp for low-σ B variants; ~0.7pp when one arm is Condition A; all 8 sig gaps exceed both."
- **Variance-collapse range conflicted between abstract (2.3–9.9×) and contribution #5 (2.0–9.9×).** Contribution #5 now reads "2.0–9.9× across the eight-model panel; 2.3–9.9× for Haiku at N=200" — population-explicit.
- **Pseudo-precision +5.81pp / +2.83pp** in body text replaced with +5.8pp / +2.9pp. Full precision retained only in Table 13.
- **CooperativeBlend2P "50.0% ± 0.000".** Now marked with † footnote: "structural identity, observed σ < 0.005pp is floating-point rounding."
- **§8 SD convention now explicit:** "every ± is sample SD across games, not SE of the mean."

## Hedging propagation

- "alignment-induced cooperation bias" was hedged in abstract but unhedged in §4.1, Related Work heading, §9 conclusion. Now: §4.1 carries the observational caveat with a §6.8 forward ref; conclusion uses "the cooperation prior" (already in abstract).
- Conclusion p.26 "scaffold-irrelevant models (Opus 4.6, Gemini Pro)" → "no detectable effect at N=50 ... N=50 resolves only effects larger than ~±5pp" (matches §4.7 exactly).
- Abstract: rewrote the trailing two paragraphs. Bias-scalar diagnostic moved earlier; closes on Middleman result; live LLMIO sentence qualified as "Haiku-vs-Haiku" with "cross-model LLMIO remains future work."

## Missing citations added

- **Lewis et al. 2017** "Deal or No Deal?" (EMNLP) — foundational neural negotiation.
- **He et al. 2018** "Decoupling Strategy and Generation in Negotiation Dialogues" (EMNLP) — direct architectural predecessor to the computed-target/centaur pattern. Added with explicit acknowledgment in §2.
- **Peters & Schaal 2007** "Reinforcement learning by reward-weighted regression" — cited in Appendix A.2 where RWR is introduced.
- **Wang et al. 2022** "ToM2C" (ICLR) — replaces the misattributed Oguntola entry.

## Reproducibility detail added

- §3.2 NegotiationGRU: encoder intermediate width = 64 stated. Parameter breakdown explicit: 2976 + 6336 + 1221 + 545 = 11,078.
- Appendix A.1 NegotiationPolicyNet: hidden = 64 with 2 LayerNorms stated. Architecture written out.
- §3.2: HeuristicNegotiator described in 2 sentences (target, concession rate, accept rule) with pointer to `heuristic_om.py`.
- Appendix A.2: r̄ defined as "global mean buyer utility over all 4,341 training tuples"; 660-game subset described (seller-first Cond B + N=50 randomized-opener ablations).
- §5.1 / §5.4: CooperativeBlend2P 50/50 algebra written out explicitly (U_A − U_B = Σ(w_a − w_b) = 0).
- Eq.4 ×5 factor: rationale spelled out ("1/E[w_i] under uniform prior, so bias has vector-dimension-independent interpretation").

## Polish

- Single-dialect (AmE) pass: behaviour→behavior, modelling→modeling, maximise→maximize, generalise→generalize, normalise→normalize, favourable→favorable, artefact→artifact. (`modelling` in Section 4.4 heading was kept as "Opponent Modeling" — title-case AmE.)
- All `\subsection{... ($N=200$) ...}` headings wrapped with `\texorpdfstring` to silence hyperref warnings.
- `\emergencystretch=3em` added to preamble — removed 5 of 7 overfull boxes globally without rewording.
- §8 `\paragraph{Models.}` paragraph wrapped in `sloppypar` to absorb long OpenRouter URL slugs.

## Build status

- 32 pages, builds clean on pdflatex+pdflatex (no bibtex needed; .bbl pre-built).
- 0 undefined references, 0 undefined citations.
- 0 hyperref Token-not-allowed warnings.
- 2 sub-pixel overfull boxes remain (lines 158, 873; ~3pt each, invisible in print).
- All 38 packet files verify against MANIFEST.csv.

## What the audit found that was already fine (no fix needed)

- All equations (1–4) and parameter counts verified correct.
- CooperativeBlend2P 50/50 algebra checks out (the formal proof now appears in the body).
- Hedging on Appendix A non-load-bearing status is consistent across abstract, body, §6.8, A.6.
- Channel-mediated account language is appropriately hedged ("consistent with...").
- Reference list venues and years are correct (modulo the fixed Oguntola/Wang swap).
- Sim-to-real GRU accuracy story (Section 4.3, 4.10) is internally consistent.

## Round 2 (second-swarm verification) — additional fixes

A second audit swarm caught three smaller issues that landed in the v1 build but contradicted the round-1 narrative. All three resolved:

- **Variance-collapse panel range corrected.** Contribution #5 and §6.1 finding (3) and §9 finding #3 previously read "2.0–9.9× across the eight-model panel" — but Table 9 shows the panel maxes out at 2.78× (Grok 3). The 9.9× value only comes from Haiku at the Oracle weight source (Table 8). Fixed to: **"1.1–2.8× across the eight-model panel; 2.3–9.9× on the Haiku ablation gradient at N=200, reaching 9.9× at the Oracle weight source"**.
- **Table 9 tier-3 label** still read "Scaffold-irrelevant" in both the caption and the in-table sub-heading even though the body text and conclusion had switched to "no detectable effect at N=50". Caption and sub-heading both updated.
- **§2 Related Work bare claim** "even accurate external intelligence cannot overcome cooperation bias when delivered as text" had no hedge. Now reads with the observational caveat and §6.8 forward-ref, and cites perez2022discovering + sharma2023sycophancy.

Also cleaned up: dead `paper.bbl` stub deleted (the bibliography is inline in paper.tex; no .bbl is needed for arXiv submission). Build re-verified clean.
