"""
arch_variants
=============
Architecture-exploration code (docs/specs/SPEC_architecture_exploration.md):

  attention.py        functional_call-safe attention, transformer blocks, MoE feed-forward (T0/T2/T4, F3, X6)
  flexnet.py          FlexNet: fusion variants (F1 mid, F2 FiLM, F3 cross-attention), transformer
                      backbones (T0, T4) and fusion encoders inside an expert bank
  metric_learners.py  episodic metric learners without an inner loop: ProtoNet (P1), Matching Net,
                      Relation Net, R2-D2 ridge head, FEAT
  adann.py            ADANN (P3): supervised pretraining with a gradient-reversal user-ID head

Every model here keeps the repo contract: forward(x_emg, x_imu=None, demographics=None,
return_routing=False), backbone() -> (feat, [feats]), get_features(), .head, .feat_dim.
"""
