# Team B replay imitation results

Revanth trained two policies to choose among legal moves by imitating recorded actions from the October 19, 2026 Kaggle replay dataset. The 24 games yielded 7,919 decisions. Both models used the same episode split, with 5,868 decisions from 18 games for training, 971 from three games for validation and 1,080 from three games for testing.

| Policy | Exported model agreement on held-out moves | Wins against heuristic-v1 | Match win rate |
| --- | ---: | ---: | ---: |
| B-SR1 linear | 478/1,080 | 228/400 | 57.0% |
| B-SR2 MLP | 518/1,080 | 243/400 | 60.75% |

Both policies played two batches of 200 best-of-three matches against Chelsi's heuristic-v1, alternating seats with the official sample deck on both sides. B-SR1 won 118 and 110 matches. B-SR2 won 124 and 119. All 800 matches finished with zero invalid responses or aborted games. Engine shuffles vary between runs.

B-SR2 was selected for Kaggle. On November 25, 2026, the Team B submissions page showed B-SR2.tar.gz as Complete under Srinidhi Jaya Revanth with a rating of 409.8. This is a dated ladder rating. Local win rates measure performance against one opponent with one deck. The three held-out episodes also limit how broadly move agreement can be generalized.

[Replay dataset](https://www.kaggle.com/datasets/kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-10-19) · [Competition submissions](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/submissions)
