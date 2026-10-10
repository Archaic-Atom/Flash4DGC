# Integration validation

Date: 2026-10-10.

## Passed locally

- Unified CLI help and three CPU entry-point tests.
- Python compilation of the integrated `flash4dgc` package and tests.
- Plane codec CPU roundtrip fixtures, including zero pages, signs/tails,
  controls 0/31/32/33, int8 endpoints and a full fine-grid-sized array.
- Result aggregation from existing frame files with frame-index, finite-time and
  300-frame quality-summary checks.
- Byte comparison confirmed that the five core codec/CUDA/state files and both
  upstream training entry points match their preserved source copies.

Commands from the repository root:

```bash
python -m unittest discover -s tests -v
python -m compileall -q flash4dgc tests
python -m flash4dgc self-test
python -m flash4dgc --help
```

The numerical codec and CUDA files were copied from the measured experiment
implementation. Integration adds repository-local entry points and portable
configuration paths. The historical source files remain in `flash_experiments`.

## Server validation boundary

The new integrated tree has not been CUDA-compiled or trained on this Windows
host. The previously returned CUDA/scene results describe the original measured
scripts, not a new benchmark of this packaging change. Before using this tree as
the replacement server checkout, run `python -m flash4dgc self-test --gpu`, then
encode and evaluate an existing frozen model using the README commands. This is
an integration check, not a request to retrain the benchmark.

The requirements file is an intended minimal Linux setup. A new clean environment
installation was not performed locally. Keep the already working server environment
for the initial integration check.

## Documents

The writing handoff and tables use Markdown, not LaTeX. The project prose scanner
is LaTeX-only. Markdown headings, tables, units, metric directions, source links
and timing-selection disclosures are reviewed in the delivered documents.
No manuscript compilation or bibliography modification is part of this delivery.
