# Frozen shared HPs for the architecture screen (`hpo_ablation_launcher.sh --fixed-params <abs path>`)

Kai (2026-10-30) holds the A4 / A2 HPO until EncoderMoE is final. The screen therefore freezes the shared
MAML training HPs at the **legacy Trial-89 values** (`make_base_config`), the same for every variant, so
variants are ranked under identical training settings while each tunes only its own HPs.

- `legacy_shared_maml.json`: outer LR, weight decay and all MAML inner-loop HPs. For the X (MoE) and
  F (fusion) tokens, and A4_ANIL / M0_ANIL.
- `legacy_inner_only.json`: inner-loop HPs only; outer LR and weight decay stay searched. For the
  transformers (T0, T2, T4), which need their own optimiser range.
- Metric learners (P1 family) and P3 use no freeze file: they have no inner loop, and their own spaces are
  already small.

**Caveat:** these values were tuned on the old 3-way env20 protocol. That is acceptable for ranking
variants, because all of them share the values. It is not acceptable for final numbers: finalists get a
full HPO (stage F). Every value is inside the current search ranges (checked by
`tests/test_arch_variants.py`).
