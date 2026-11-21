# Group 9 - GenAI Pokémon TCG agents

DATA 266, Fall 2026. One shared repository for all eight members, two Kaggle submission tracks, one combined report and presentation.

**Status: starter scaffold.** The simulator, feature encoders, baseline, models, training, evaluation and submission packaging still need implementation. Placeholder Python functions deliberately raise `NotImplementedError`; nothing here is a trained or Kaggle-ready agent.

## Start here

1. Read [the team workflow](CONTRIBUTING.md) and [seven-day execution plan](docs/execution_plan.md).
2. Find your responsibilities in your team's README and your model-folder README.
3. Setup owners reproduce the current official SDK example and share real observations within their team.
4. Define the feature schema, target, dataset split and full response handler before fitting models.
5. Track actual experiments, official submissions and contributions in the provided CSVs. Keep unmeasured values empty.

| Team | Kaggle name | Members |
| --- | --- | --- |
| A | GROUP_PROGRAMMING_09_a | Rishitha Gogineni, Kavya Ayyappan, Shashira Guntuka, Jiayi Liang |
| B | GROUP_PROGRAMMING_09_b | Chelsi Shulamit Elthuri, Sadaf Fatima Syeda, Bhavyasree Kondi, Srinidhi Jaya Revanth Srirangarajapally |

## Repository map

```text
team_a/                    Team A source, configs, models and experiment metadata
team_b/                    Team B source, configs, models and experiment metadata
docs/execution_plan.md     Daily gates and ownership
docs/requirements.md       Course requirements versus planning assumptions
docs/contributions.csv     All eight members; planned and completed work separated
docs/decision_register.csv Open questions and decisions
docs/report/               Combined academic report outline
docs/slides/               Combined presentation outline
evidence/team_a/           Instructions for genuine Team A evidence
evidence/team_b/           Instructions for genuine Team B evidence
.github/                   Pull request template
```

Each team contains `src/`, `configs/`, `scripts/`, `tests/`, `data/`, `experiments/`, `submissions/` and `agent_package/`. Keep local datasets, generated checkpoints, logs, decks and archives out of Git by default; see [.gitignore](.gitignore).

## Runtime and commands

No packages are installed by this scaffold. The starter interfaces use Python 3.10+ syntax; select a Python version compatible with the actual SDK before running or pinning dependencies. Each team records its verified runtime in `configs/environment.json` and its packages in `requirements.txt`.

Training, evaluation and packaging scripts are explicit placeholders. There are no verified training or Kaggle submission commands yet. The first working baseline commit should add exact reproduction commands to the team's README.

## Shared repository decision

All eight members use this one repository, with `team_a/` and `team_b/` for organization. Folders and branches do not limit who can read the code. The competition's applicable sharing rules still need verification; the shared layout is a group decision, not evidence of organizer permission. Repository visibility and collaborator permissions have not been changed by the scaffold.

Official references: [competition](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground), [rules](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/rules), [cabt documentation](https://matsuoinstitute.github.io/cabt/).
