# Blueprint Review Viewer

This is a read-only local reviewer for Blueprint experiments.  It
binds to `127.0.0.1` only and accepts no write HTTP methods.

For a legacy run, create review artifacts once (this writes only `review.json`
metadata beside the existing candidates):

```bash
python experiments/blueprint_review_viewer/backfill.py EXPERIMENT_NAME
```

Start it on the remote machine:

```bash
experiments/blueprint_review_viewer/run_blueprint_review_viewer.sh
```

The server scans `/ssd/czx/czx_work/cot_blueprint_refine` by default. The
browser search lists only experiment directories containing
`robustpa/blueprint/results.jsonl`; paste a complete or partial `exp_name`, then
click a match to load it. If the experiment output base is elsewhere, set
`BLUEPRINT_REVIEW_OUTPUT_BASE`.
The optional `BLUEPRINT_REVIEW_SSH_TARGET` environment variable controls the
target shown in the tunnel command.

Then run the printed command on your local machine:

```bash
ssh -N -L 8766:127.0.0.1:8766 user@remote-host
```

Open `http://127.0.0.1:8766`. The viewer reads the current Whole-COT
`generation_history` schema. Selecting `Generation N` shows that round's two
feedback classes:

- deterministic validation, split into whole-graph/Lean validation and the
  Phase 2 standalone preflight;
- semantic validation from the Formal Decompiler and Whole-COT Comparator.

The right pane also has four large, collapsed-by-default sections for the
selected generation: Decompile response, Compact response, Builder input, and
Builder response. Think and non-think output are separated. Builder input is
reconstructed from the previous generation's persisted Blueprint and its
deterministic/semantic errors. The Builder non-think panel displays the
persisted, canonicalized generation Blueprint because raw tool-call arguments
are redacted in current traces.

The viewer does not interpret the removed Phase 1A/1B edit lifecycle format.
Existing older-schema `review.json` files are rebuilt in memory from
`results.jsonl`; running backfill first is not required.
