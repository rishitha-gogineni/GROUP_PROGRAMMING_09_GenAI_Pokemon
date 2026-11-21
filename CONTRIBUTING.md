# Team workflow

## Ownership

Everyone collaborates in this single repository. Keep team-specific code in its own team directory and put your model implementations in your named folder. Shared adapters and encoders are shared within each team.

Use your own GitHub identity for commits. Update `docs/contributions.csv` with actual work and links to commits/PRs, runs and official submissions. Folder ownership alone does not establish a completed contribution.

## Everyday Git workflow

These commands are instructions for the team, not commands executed by this document.

1. Clone this repository once, then open its directory.
2. Start from an updated `main`: `git switch main`, then `git pull --ff-only`.
3. Create a short branch, for example `git switch -c feature/rishitha-baseline`.
4. Make a focused change. Review `git diff` and `git status`.
5. Stage specific intended files; avoid automatically staging data, credentials or generated artifacts.
6. Commit with a clear description and push your branch: `git push -u origin HEAD`.
7. Open a pull request describing the change, validation and remaining work. Have a teammate review it before merging when practical.
8. Update your local `main` after the merge.

Do not force-push a shared branch. Avoid multiple people editing the agent entry point simultaneously. Ask the team's integration owner to coordinate interface changes.

## Installation and execution

Do not install packages or execute project programs without the responsible user's permission. This scaffold provides no automatic setup, training, CI workflow or submission actions. Reading or editing source is separate from running it. Agree on a compatible runtime, then document the exact permitted commands and versions.

## What a model contribution includes

- Architecture, input/output schema and supported decision contexts.
- Training data source, target, objective and match-grouped split.
- Actual implementation, artifact/config identity and measured evaluation.
- Kaggle attempt/status/evidence as required by the course.
- Limitations and an honest account of ineffective or incomplete work.

Two architectures per person are planned pending TA interpretation of the assignment's 2-3 requirement. A different seed, learning rate or filename alone does not establish a distinct architecture.

## Data and academic work

Use only information available to the acting player as model inputs. Separate training, validation and held-out evaluation games. Preserve raw counts and genuine evidence; do not replace missing scores with estimates.

Use `docs/report/` and `docs/slides/` for the combined deliverables. Verify applicable competition sharing rules before distributing code, decks, card data, replays or evidence. Do not add a public license to competition-supplied material without checking its terms.
