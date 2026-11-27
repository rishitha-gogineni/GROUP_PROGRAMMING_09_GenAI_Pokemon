# Team A experiment index

Find experiments by owner. Start with each member's README rather than individual log files.

| Member | Results | Model explanation |
| --- | --- | --- |
| Rishitha | [Baselines, model training, common evaluation and energy experiment](rishitha/) | [A-R1 and A-R2](../src/models/rishitha/) |
| Kavya | [Saved checkpoints](kavya/) | [A-K1 and A-K2](../src/models/kavya/) |
| Jiayi | [Existing policy evaluation CSV](../src/models/jiayi/evaluation_results.csv) | [A-J1 and A-J2](../src/models/jiayi/) |
| Shashira | Add completed results under `shashira/` when available | [Model work folder](../src/models/shashira/) |

[results.csv](results.csv) is the shared experiment log and [model_registry.csv](model_registry.csv) links model identities to artifacts. Member rows may lag behind work in model folders; reconcile them against evidence before the report. A planned row does not prove a model is unfinished.

New common evaluation exports should go under the owner's folder with separate model IDs and run IDs. Retain executed notebooks and comparison CSV/JSON. Record protocol and runtime fingerprints, denominators, errors and uncertainty. Keep individual battles, externally controlled matches and hosted Kaggle ratings separate. Do not mix their percentages or overwrite earlier observations.

Original reports and source snapshots retain their historical runtime paths. Rishitha's [cleanup manifest](rishitha/cleanup_manifest.json) maps the repository moves and verified duplicate removal. Original source, measured outcomes, training data and failed experiments remain traceable.
