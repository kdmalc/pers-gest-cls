# docs — index

**Venue:** ICML (Jan) or TMLR, decided after the main runs finish.

## Main docs (start here)
| File | What |
|---|---|
| `REBUTTAL_CHECKLIST.md` | Every run that goes in the paper (or decides what does), with status and accuracy |
| `CURRENT_TODOs.md` | Open threads, decisions needed, and the verbatim cluster commands |
| `PLAN_resubmission.md` | Big picture: reviews → experiments → stages → venue |
| `LOG.md` | Short history of decisions and finished work |

## Subfolders
| Folder | What goes there |
|---|---|
| `specs/` | Design specs and pre-registered rules: `SPEC_*.md` (protocol, HPO, stats, episodes, 2 kHz EMG, architecture exploration, …), `DECISION_method.md` (go/no-go), `SOTA_COMPARISONS.md` |
| `analysis/` | Analyses and idea notes: `COVARIANCE_TUTORIAL.md` (why Riemannian MDM works, reading list), MoEMeta comparison, eval-pipeline audit, ability-cluster hypothesis, study plan, extra ideas |
| `run_reports/` | Write-ups of specific cluster runs (debug runs, diagnostics) |
| `papers/` | PDFs: our NeurIPS'26 submission, MoEMeta, the OpenReview thread. Local only, never committed (git-ignored). |
| `archive/` | Superseded docs: the old full running log, old status / TODO / guide files, old plans. Read with care; dated. |
