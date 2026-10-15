# Contributing

Issues and pull requests are welcome.

- Set up a development install with `pip install -e ".[dev]"` and `pre-commit install`.
- Keep the algorithm pieces in `src/mutual_taught/algorithms/` free of torch.
- Run `ruff check src && ruff format --check src` before opening a pull request; CI runs the same
  checks.
- When an implementation detail is not specified by the paper, document the choice next to the
  code that implements it.
