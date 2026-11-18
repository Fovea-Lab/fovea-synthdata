> **AI authorship:** Created by OpenAI Codex. **Model:** the exact runtime model identifier was not exposed to the assistant.

# Contributing to Fovea SynthData

An example with a known answer is a good starting point for a contribution. It lets another developer reproduce a failure, decide whether a new generator is needed and test a proposed fix.

## Report a problem or propose a generator

For a bug, include the installed version, generator, preset or minimal YAML, seed, writer, exact command and the observed and expected results. For a failed recoverability check, include the relevant row of `check/report.json`. A compact synthetic example is enough.

For a new generator, describe the pipeline input, the output you want to measure, how the generator would record the correct answer, and a small case that would demonstrate recovery. If one of the current generators already supplies the right objects and truth tables, a new preset may cover the case. [Open an issue](https://github.com/Fovea-Lab/fovea-synthdata/issues/new/choose)

## Develop locally

Use Python 3.12+ and uv. From the repository root, run the default suite against the locked dependencies.

```sh
uv sync --locked --all-extras
uv run pytest -q
```

For a change that affects the full-plate renderer, the reference measurements or the offline report, run the larger fixtures too.

```sh
uv run pytest -q -m slow
```

## Add a registered generator

1. Create a module under `src/synthdata/generators/` with a Pydantic design model, a small bundled YAML preset and a generator class. Implement the generator protocol's design loading and serialization, preview, contact sheet and dataset generation. Expose the generator name, version, available writers, default writer and preset list. The built-in particle generator provides a worked example. [Generator and writer protocols](src/synthdata/core/generator.py) · [Particle generator](src/synthdata/generators/particle_plate/generator.py)
2. Add the built-in name and import target to the registry. Keep catalog discovery lightweight by loading rendering and format dependencies when their operations run. Add generator-specific dependencies to the matching optional extra and confirm that the installed wheel includes the preset. [Registry](src/synthdata/core/registry.py) · [Packaging configuration](pyproject.toml)
3. Follow the dataset contract. Use a stable seed for each generated unit, give measurements and truth a common object or field identifier, use `prepare_output` for destination handling, and write the manifest after the writers have closed. Document the inputs a measurement pipeline can consume independently of the answer key. [Manifest writer](src/synthdata/core/provenance.py)
4. Supply a known-answer test. Render the small preset twice and compare the output checksums, read back the advertised formats, and assert at least one measurable result against ground truth. Exercise `synthdata describe`, preset export and generation through the installed CLI. If the generator implements a recoverability check, exercise that operation and its failure result as well.
5. Add a catalog page with an example field, its mask or other truth artifact, the matching identifier, and a measured result. Link the page from the README. Include settings, default output format and any feature the reference pipeline leaves unmeasured.

## Submit a pull request

Describe the pipeline case the change covers and link the issue if one exists. Include the short command that reproduces the result, the checks you ran and one expected value with its units. For changes to an existing renderer, call out any change to seeded images, exported columns, labels or file paths so users can assess the effect on their fixtures.
