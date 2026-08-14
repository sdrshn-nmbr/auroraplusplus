# Aurora++

Aurora++ is an experiment in safe, continual speculative decoding.

A large target model writes the final output. A smaller draft model guesses the next
tokens. The target model checks each guess. Only target-approved tokens leave the
service.

Aurora++ learns from recent requests and updates the draft model. It accepts an update
only when the update keeps the target behavior and improves measured performance. If an
update fails, the service keeps or restores the last good draft model.

## Goal

Build one system that can:

1. Serve requests with a fixed target model.
2. Record complete training evidence.
3. Train the draft model from that evidence.
4. Save and reload a new draft model.
5. Compare the new model with its parent.
6. Promote a faster, safe model.
7. Roll back a bad model without manual repair.
8. Explain every decision from stored evidence.

The first runtime uses:

- Target: `poolside/Laguna-XS-2.1-INT4`
- Draft model: `poolside/Laguna-XS-2.1-DFlash-INT4`
- Trainer: SpecForge
- Server: SGLang
- Hardware: two Modal H100 GPUs
- Control database: PostgreSQL
- Training data: approved rows from `GPUMODE/KernelBook`

The exact model, code, data, runtime, and evaluation revisions are in
[`configs/laguna_dflash_int4.json`](configs/laguna_dflash_int4.json).

## Current state

The model path works. The complete live service does not run yet.

Physical H100 probes have shown that the current code can:

- Load the pinned Laguna target and official DFlash draft model.
- Capture the target data that DFlash training needs.
- Give a captured batch to SpecForge.
- Run a real optimizer step and change the draft model.
- Save the weights, optimizer state, random state, and training position.
- Reload the saved model in a new process.
- Serve the reloaded model with SGLang.
- Restore and serve the parent model.
- Match target-only output exactly during greedy decoding.
- Pass the signed sampled-output comparison.
- Replay one captured batch with the same hashes.

The sampled-output test does not require the same token for the same seed. The target
runtime itself did not repeat that result reliably. Instead, the test first measures the
target's normal variation. It then checks that adding DFlash does not add more variation
than the signed limit. Exact greedy output remains a hard requirement.

The repository has 108 passing tests. They cover typed contracts, signatures, the data
firewall, the controller, promotion rules, fault simulation, model conversion, training,
and compatibility evidence.

The system remains off because these end-to-end proofs are still missing:

- A complete signed data manifest with physical row checks.
- One signed experiment that binds all code, runtime, data, and evidence.
- A durable shadow deployment that survives restarts.
- The full paired speed test at concurrency 1 and 4.
- One real canary promotion and one forced rollback.
- The live judge and human correction loop.
- The one-way public benchmark audits.

This distinction is important: the core model port works, but Aurora++ is not yet a
self-improving production service.

## Safety rules

Aurora++ fails closed.

- The target model stays fixed during a draft-model experiment.
- Automatic control can promote or roll back only the draft model.
- A human must sign changes to hardware, runtime, data, target, tests, limits, or workload.
- PostgreSQL is the control record.
- Large evidence files use immutable, content-based paths.
- A failed learner or judge cannot stop the last good serving path.
- An unsafe serving path falls back to target-only decoding.
- KernelBench, KernelBenchX, and KernelBench-Hard do not enter normal training or model
  selection.
- The judge can suggest a change. It cannot approve its own change.

## Control states

The controller has six states:

- `OFF`: no GPU work should run.
- `STARTING`: workers load and prove their identity.
- `SHADOW`: capture, training, and evaluation run without automatic promotion.
- `AUTO_DRAFTER`: signed rules allow draft-model promotion and rollback.
- `DRAINING`: new work stops while current work ends.
- `FAILED`: an invariant failed; serving stays on the last good model.

An uncertain restart returns to `SHADOW`.

## Local setup

Aurora++ requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run aurorapp config check
uv run aurorapp --help
```

Run the local checks:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
```

Generate the JSON Schemas from the Python types:

```bash
uv run aurorapp config schemas
```

The GPU probes require Modal access and the pinned model artifacts. A local passing test
is not a physical GPU result. A physical GPU result is not a live service result. Reports
keep these evidence levels separate.

## Repository map

- `src/aurorapp/models.py`: typed source of truth.
- `src/aurorapp/controller.py`: fail-closed state changes.
- `src/aurorapp/database.py`: PostgreSQL control record.
- `src/aurorapp/compatibility.py`: model and runtime gate.
- `src/aurorapp/training_probe.py`: bounded training proof.
- `src/aurorapp/oracles.py`: correctness, output, performance, and health checks.
- `src/aurorapp/promotion.py`: paired promotion rules.
- `src/aurorapp/simulation.py`: repeatable failure tests.
- `src/aurorapp/judge.py`: Codex judge boundary.
- `configs/`: checked experiment input.
- `schemas/`: generated JSON Schemas.
- `artifacts/`: checked evidence and signed contracts.
- `tests/acceptance/`: public behavior and safety tests.

## Claims and non-claims

Aurora++ tests three separate ideas:

1. A trained draft model can improve speed without changing target behavior.
2. A continual service can train, evaluate, promote, recover, and roll back safely.
3. Human corrections can improve the evaluator without giving it control.

The repository does not yet prove any complete claim above.

It also does not claim that speculative decoding improves answer quality, that DFlash is
better than Aurora, or that the public target never saw related benchmark data during
pretraining. Aurora is a later control and a design reference. It is not the first runtime.
