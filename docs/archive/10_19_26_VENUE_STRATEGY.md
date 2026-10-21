# VENUE_STRATEGY.md — where to send this, and how to frame it for each

## The strategic read

Four reviewers scored the paper 3/3/3/3 and, critically, split their assessment
the same way every time: **originality 2/2/1, but R2 and R4 credited the
evaluation rigor unprompted.** That is a very specific signal. It says the paper
is not weak — it is *mis-venued*. We submitted an empirical-characterisation
paper about an under-served population to a venue whose primary currency is
methodological novelty, and it was correctly judged against that currency.

Three consequences shape everything below.

1. **Resubmitting the same framing to another top ML conference is the
   highest-risk path available.** An originality score of 1 is not a
   presentation problem; MoE + meta-learning has a NeurIPS 2025 precedent
   (MoEMeta), and no amount of rewriting makes the pairing novel. We would be
   re-running the same experiment on a new set of reviewers.
2. **Our actual assets are venue-mismatched, not weak.** 32 participants of whom
   26 have motor impairments; user-defined gesture semantics; a leave-two-subjects-out
   protocol over all 32; episode-level paired comparisons; a genuinely hard
   deployment regime. At an HCI/wearables or neural-engineering venue those are
   the *contribution*, not the setup.
3. **A journal-track or rolling venue is worth more to us than a conference
   deadline.** Our reviewers were consistent and specific. A revise-and-resubmit
   process converts that into an iterative conversation; a conference converts it
   into another binary coin flip with fresh reviewers who will find fresh
   objections.

Everything below assumes the Phase 0/1 work in the plan lands — the fusion
redesign, the prototype head, real transformer baselines, honest HPO, and the
variance decomposition. Without that, no venue choice saves the paper; with it,
several do.

---

## Recommendation

**Target ACM IMWUT (UbiComp) as the primary venue.** It is the only option where
the timing, the review model, and the evaluation criteria all line up with what
we have and when we will have it.

| | IMWUT |
|---|---|
| **Timing** | Rolling quarterly cycles (typically mid-Feb / mid-May / mid-Aug / mid-Nov — **verify current dates**). The next one lands roughly 2.5 months out, which is exactly the plan's 6–7 week experimental runway plus writing time. Every other top venue is either ~3 weeks away (impossible) or ~9 months away (wasteful). |
| **Review model** | Journal-track, with a real **major-revision** path. Our reviewers' objections were specific and mostly answerable; a revision cycle rewards that, a conference reject does not. |
| **What it wants** | Novel *systems and empirical understanding* in ubiquitous/wearable sensing. Multimodal wearable sensing on a clinical population with a rigorous cross-subject protocol is squarely central, not peripheral. |
| **Novelty bar** | Contribution can be the system, the dataset, the empirical characterisation, or the deployment insight. "No individual component is new" is not disqualifying if the composition and the finding are. |
| **Risk** | Reviewers will push hard on *deployment realism*: real-time latency, on-device feasibility, single-session limitation, and whether 1-shot 3-way is a real interaction or a lab construct. Those are answerable, and some (latency, model size) are cheap wins we currently do not report at all. |

**Framing for IMWUT.** Lead with the interaction problem, not the architecture:
*people with motor impairments cannot use fixed gesture vocabularies, so the
system must learn a user's self-defined gestures from a single demonstration.*
Then: what does it actually take to make that work at N=32, what does each
modality contribute, where does it break as the vocabulary grows, and what is the
honest ceiling? The MoE becomes an implementation detail that earned its place
via ablation — which is exactly the claim our data supports.

**The one thing that changes this recommendation:** if the dataset can be
publicly released, **NeurIPS Datasets & Benchmarks becomes co-equal or better**
(see Goal 2). Settle the release question early — it is the highest-leverage
open decision in the whole strategy, and it is a permissions/IRB question, not a
research one.

---

## Goal venues

### Goal 1 — ACM IMWUT / UbiComp ★ recommended

Covered above. Concrete additions needed beyond the plan: inference latency and
model size on a plausible target device; a short discussion of the enrollment
interaction (how a user actually demonstrates a gesture once); and the per-gesture
breakdown, since an HCI audience will want to know *which* gestures fail, not
just the mean.

### Goal 2 — NeurIPS Datasets & Benchmarks track

**Conditional on being able to release the data.** If we can, this is arguably
the strongest option in the set.

- **Why it fits.** The D&B track explicitly values datasets, benchmarks and
  rigorous empirical studies. It does *not* apply the main track's novelty bar.
  Our meta-review fallback position — "weigh the problem formulation and the
  empirical characterisation of a regime for which no comparable dataset exists"
  — is not a fallback at D&B; it is the submission category. The track also
  already takes sEMG seriously (emg2qwerty, emg2pose).
- **Why it is strong for us specifically.** It converts our biggest liability
  into the asset: *no comparable dataset exists for this population and this
  task*, and we have one, with 26 motor-impaired participants and user-defined
  gesture semantics that no able-bodied fixed-vocabulary corpus can instantiate.
- **Framing.** The dataset and the benchmark protocol are the contribution.
  EncoderMoE becomes one entry in a baseline suite that also contains
  PCA+KNN, Riemannian tangent space, ROCKET, ProtoNet, a properly-sized
  transformer, and the Kaifosh transfer study. **PCA+KNN beating our model stops
  being a problem and becomes a headline finding**: "on this benchmark, a
  nonparametric baseline is competitive with meta-learned adaptation at low
  way-count, and the gap only opens at 10-way" is a genuinely useful result that
  shapes what others build.
- **Cost.** A D&B submission needs a documented data card, licensing, consent and
  de-identification review, hosting, and a maintenance commitment. Non-trivial,
  and IRB-dependent for a clinical population. Timing: NeurIPS 2027 cycle,
  roughly 8–9 months out — so this is the patient option, and it can run in
  parallel with an IMWUT submission of the *methods* paper if the two are scoped
  cleanly apart.
- **Risk.** If the data cannot be released, this option disappears entirely.
  Check first.

### Goal 3 — IEEE TNSRE (Transactions on Neural Systems and Rehabilitation Engineering)

- **Why it fits.** The natural disciplinary home: EMG/IMU gesture recognition,
  assistive interfaces, clinical populations. Côté-Allard (our [7]) is published
  there, so our closest baseline shares the venue and the reviewers will know it
  cold. Rolling submission, no deadline pressure, real revision cycles.
- **What it wants.** Clinical/translational relevance, methodological soundness,
  and an honest treatment of the gap between lab and use. Architectural novelty
  is close to irrelevant; *population* and *protocol* are what count.
- **Framing.** Lead with the population and the clinical motivation. Our 26
  motor-impaired participants are the strongest thing in the paper here. Cite
  Scheme & Englehart on the clinical-translation gap in the introduction, and
  frame the single-session and 10-repetition limits as participant-burden
  constraints in this population — which is true and which this readership will
  immediately accept, where an ML audience read it as a design oversight.
- **Risk.** Reviewers will expect comparison against the *classical* myoelectric
  control literature: TD features, LDA, and Riemannian methods. That is exactly
  the baseline gap we already have (D2), so the work is shared with the plan.
  They will also ask about real-time control and window latency, which our 3.2 s
  window makes awkward — prepare an answer.

---

## Fallbacks

### Fallback 1 — TMLR (Transactions on Machine Learning Research)

Honestly close to a goal venue for our situation, and listed here only because
its prestige is still accruing relative to NeurIPS/ICML.

- **Why it fits so well.** TMLR's acceptance criteria are explicitly *(a) are the
  claims supported by convincing evidence, and (b) would some subset of the
  community be interested* — **novelty is deliberately not a criterion.** Our
  reviewers attacked novelty and credited evidence. That is the exact profile
  TMLR was designed to accept.
- **Practical advantages.** Rolling submission, no deadline, fast decisions
  relative to a journal, action-editor-managed revisions, and no page-limit
  pressure on the ablation tables and appendices this paper needs.
- **Framing.** Keep the ML framing, but rewrite every claim to be exactly as
  strong as the evidence supports: "routed encoder capacity plus meta-learned
  adaptation are jointly necessary in this regime, with parameter-matched
  controls" rather than "we propose a novel architecture." Include the negative
  results — the ensembling-not-specialisation routing finding, PCA+KNN's
  competitiveness, the fusion result whichever way it lands. TMLR rewards that
  where a conference punishes it.
- **Use it if:** the HCI framing feels forced, or IMWUT's next cycle is missed.

### Fallback 2 — ASSETS 2027

- **Why it fits.** The premier accessibility computing venue, and this paper is
  accessibility research that happens to use ML. A 26-participant motor-impaired
  study with user-defined gestures is a strong ASSETS contribution.
- **Framing.** Ability-based design (Wobbrock et al.). The user-defined gesture
  vocabulary *is* the accessibility contribution: the system adapts to the
  person's actual movement capability rather than requiring a canonical gesture.
  Foreground participant experience and the enrollment burden — one demonstration
  per gesture — as the design constraint that motivates the 1-shot requirement.
- **Trade-off.** Smaller and more specialised than CHI or IMWUT, and the ML
  contribution will be read as a means rather than a result. Deadline typically
  spring 2027 — verify.
- **Note:** our dataset originates at UIST 2024, so **UIST 2027** is a
  credible sibling option with a similar framing and a spring deadline.

### Fallback 3 — Journal of Neural Engineering, or IEEE JBHI

- **JNE** and **IEEE JBHI** are both solid, rolling, well-regarded domain
  journals with essentially the TNSRE framing at a slightly lower bar. JBHI
  skews more toward health informatics and would want the clinical-population
  angle foregrounded even harder.
- **Use them if:** TNSRE rejects, or if reviewers there demand real-time control
  experiments we cannot run.
- Avoid the low-selectivity megajournals (Scientific Reports, PLOS ONE) unless
  the goal has become "get it published and move on." The paper is better than
  that, and this population deserves a readership that will act on the result.

---

## On NeurIPS, ICML and ICLR specifically

You asked about these directly, so here is the unvarnished version.

**The main tracks are a bad bet for this paper as currently scoped.** With
originality scored 2/2/1 and a NeurIPS 2025 precedent for the core pairing, the
novelty objection is structural. Rewriting the contribution statement does not
remove it, and reviewers at these venues are specifically selected to weigh it.

**The conditions under which a main-track resubmission becomes reasonable** — all
of them, not some:

1. Phase 1 delivers a **method result that is new, not just better**. The two
   candidates in the plan are (a) the modality-competition finding with a fix
   that makes EMG contribute where naive fusion cannot, and (b)
   task-conditional routing (`MOE_task_routing_mode="add"`) beating both
   query-only and support-only routing, which would turn R3's novelty attack into
   a positive empirical result.
2. The transformer comparison is **param-matched with an equal, stated HPO
   budget**, and the naive-config row is shown next to it.
3. The model **beats PCA+KNN, Riemannian tangent space, and ROCKET** — or, if it
   does not, the paper's claim is honestly restated to something it does support.
4. Effects are **larger than the variance**, with 3 seeds × 32 folds and paired
   CIs. The current 5-point M0-vs-A4 gap against a ~3-point same-config spread
   would not survive a careful reviewer.

If all four hold, **ICLR is the best of the three** for us: the open-review
process makes it far easier to pre-empt a repeat of the MoEMeta objection in the
paper itself, and rebuttal-driven score movement is more common there. ICML would
be second. NeurIPS is the worst option of the three specifically because a
resubmission may reach reviewers who saw it, and the meta-review's inverted
premise about our 3-way/10-way numbers may persist in institutional memory.

**Also worth knowing:** ICLR 2027 and CHI 2027 deadlines both fall roughly 3–4
weeks from now. Neither is achievable alongside the plan's experimental work.
Do not compress the plan to hit them — a rushed submission with the same
weaknesses is strictly worse than a considered one a cycle later.

---

## The same work, five framings

The experiments barely change across venues. The *claim* changes a great deal.
This table is the practical output of this document.

| | Headline claim | Lead contribution | MoE's role | How PCA+KNN winning is handled | 10-way weakness is |
|---|---|---|---|---|---|
| **IMWUT** | One demonstration is enough to recognise a user's self-defined gestures on a wearable | The system and its empirical characterisation | An implementation detail that earned its place by ablation | A reported baseline; we discuss amortisation and per-user storage cost | A characterised scaling limit with a stated cause |
| **NeurIPS D&B** | Here is the benchmark for 1-shot user-defined gesture recognition in a motor-impaired population | The dataset and protocol | One baseline among many | **A headline finding** about the regime | A property of the benchmark that makes it worth having |
| **TNSRE** | Cross-subject myoelectric gesture recognition is feasible with one calibration repetition in a motor-impaired cohort | Clinical relevance and protocol rigor | Method detail | Expected — classical methods are strong at small N; we quantify when | A translational constraint on vocabulary size |
| **TMLR** | Routed capacity and meta-learned adaptation are jointly necessary in this regime; routing's benefit is ensembling, and it concentrates at K=1 | Evidence quality and honest negative results | Under direct study | Reported straight, with the hybrid bound as the constructive answer | A measured limit with a mechanistic account |
| **ICLR** (conditional) | Modality competition explains why naive EMG+IMU fusion fails, and task-conditional routing fixes 1-shot high-way adaptation | The method result | The contribution | We beat it, or we do not submit here | Solved, or the submission is not ready |

Two things are constant across all five, and both are non-negotiable: the
protocol is named in every table caption (E5), and no claim is made that is
smaller than the measured variance (B1).

---

## Decision sequence

1. **Now — settle the data-release question.** Ask the PI and check the IRB
   consent language. It decides whether Goal 2 exists, and it is the single
   highest-leverage open item.
2. **Now — confirm the IMWUT cycle dates** and pick the target cycle. Work
   backwards: submission minus 2 weeks for writing, minus the plan's 6–7 weeks.
3. **Week 1 — read §0 of `READING_LIST.md`.** Four papers, and they change what
   we claim in four places.
4. **After Phase 1 reads out — choose the framing**, using the table above. The
   A22 (trained unimodal), A17 (prototype head), and PCA+KNN/Riemannian/ROCKET
   baseline results are what decide it. Do not choose earlier; the framing should
   follow the evidence, which is the mistake the first submission made.
5. **Submit to one venue at a time.** Simultaneous submission is prohibited at
   all of these, and the revision feedback from a journal-track venue is worth
   more than a parallel lottery ticket.

## Open questions for the PI

1. **Can the dataset be released publicly?** Consent language, IRB amendment
   feasibility, de-identification requirements, and institutional appetite.
   Gates Goal 2.
2. **Is an HCI/accessibility venue acceptable as the primary target**, or is
   there a requirement for a core-ML venue? This is a career/lab-strategy
   question, not a research one, and it should be answered before we write.
3. **Is there appetite for a two-paper split** — a benchmark/dataset paper and a
   methods paper — given that the material plausibly supports both and they suit
   different venues?
4. The four items in `LIMITATIONS.md` §G still need sign-off, and two of them
   (the `$B` reporting decision and the Kaifosh reframe) change the paper's
   framing regardless of venue.
