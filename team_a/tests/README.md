# Team A checks

Run the shared-tool tests from the repository root:

```bash
python -m unittest discover -s team_a/tests -p test_common_tools.py -v
```

The eleven tests use the standard library. They check the shared encoder against the frozen features and both fitted scorers on three saved observations. Temporary CLI tests use a tiny pure-Python SDK fixture to check loading without __file__, seat/outcome mapping, archive contents, explicit assets, invalid actions, changed package files, failed validation, interrupted workers, and output reuse protection. No training, installation or native Pokemon simulator is run. No real game results are produced by these fixtures.

common_tools_validation.json records this limited verification. Original Colab reports under experiments/ remain the evidence for actual native battles and model fitting. Run the documented evaluate.py and package_agent.py commands in the verified Linux SDK environment when an authorized native validation run is needed.
