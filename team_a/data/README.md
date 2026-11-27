# Local data and provenance

Use raw/ for local inputs and processed/ for derived datasets when needed. Create these local working folders before saving files; empty placeholder folders are not stored in Git. Their contents remain ignored. Do not commit competition downloads, card copies, decks or private replays automatically.

Inspect source-table grain before deduplication: card IDs may occur on multiple attack rows. Validate types, missingness, joins, unknown categories and acting-player visibility. Split by full match/game before sampling decisions and fit normalization only on training data.

Record source, permission/provenance, teacher version, collection config, game IDs, schema version, target and limitations. Use configs/data_contract.template.json per model/run. If replays are unavailable, use own-team heuristic demonstrations and describe imitation as such. Unchosen actions are not proven bad; teacher scores are not true action values.
