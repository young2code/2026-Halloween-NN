# Contributing

Small, inspectable changes are welcome. Useful contributions include new
original procedural features, accessibility improvements, browser performance
measurements, tests, and clearer educational explanations.

Before opening a change:

1. Run `npm test` and `npm run build`.
2. Run `python -m pytest -q` and `ruff check training export tests` in the
   training environment.
3. Document the provenance and license of any visual data.
4. Do not add telemetry, API keys, server inference, or copied artwork.

Please keep the deployed generator compact and preserve deterministic seeds.

