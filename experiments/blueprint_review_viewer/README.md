# Blueprint Review Viewer

This is a read-only local reviewer for a completed Blueprint experiment.  It
binds to `127.0.0.1` only and accepts no write HTTP methods.

For a legacy run, create review artifacts once (this writes only `review.json`
metadata beside the existing candidates):

```bash
python experiments/blueprint_review_viewer/backfill.py /path/to/robustpa/blueprint
```

Start it on the remote machine:

```bash
experiments/blueprint_review_viewer/run_blueprint_review_viewer.sh /path/to/robustpa/blueprint user@remote-host
```

Then run the printed command on your local machine:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@remote-host
```

Open `http://127.0.0.1:8765`.  The artifact schema includes generic operation
types (`nodeEdit`, `dependencyEdit`, `repairBundle`, `subgraphEdit`) so later
pipeline changes do not require a viewer protocol break.
