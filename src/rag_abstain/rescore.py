"""Re-classify a finished run offline (DESIGN section 9): same answers, another classifier.

`rescore` reads the latest record for every (qid, repeat) in a run directory, resolves each
stored response again with the chosen classifier and the system's markers from the manifest,
and writes the result as a run directory of its own inside the original, `rescore_<name>/`,
with its manifest, records, summary and report. No system is called, and scores are kept as
they are, since they depend only on the answer. `compare` works between the two directories.
"""

from pathlib import Path
from typing import Any

from rag_abstain.classify import make_classifier, resolve_outcome
from rag_abstain.report import write_reports
from rag_abstain.runner import MANIFEST_FILE, RECORDS_FILE, RunError, read_manifest, read_records

RESCORE_PREFIX = "rescore_"


def rescore(run_dir: str | Path, classifier_name: str) -> tuple[Path, dict[str, Any]]:
    """Re-classify every record of `run_dir` and write `rescore_<classifier_name>/` inside it.

    Returns the new directory and its summary. Re-running overwrites the previous rescore.
    Raises RunError for a directory that is itself a rescore, ValueError for an unknown name.
    """
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    if manifest.rescored_from is not None:
        raise RunError(f"{run_dir} is already a rescore of {manifest.rescored_from}")
    system = manifest.system
    classifier = make_classifier(
        classifier_name,
        extra_markers=system.get("extra_markers") or (),
        refusal_text=system.get("refusal_text"),
    )

    out = run_dir / f"{RESCORE_PREFIX}{classifier_name}"
    out.mkdir(exist_ok=True)
    rescored = manifest.model_copy(
        update={
            "run_id": f"{manifest.run_id}+{classifier_name}",
            "classifier": classifier.config(),
            "rescored_from": manifest.run_id,
        }
    )
    (out / MANIFEST_FILE).write_text(rescored.model_dump_json(indent=2) + "\n", encoding="utf-8")

    lines = []
    for record in read_records(run_dir):
        outcome, stage = resolve_outcome(record.response, classifier)
        lines.append(record.model_copy(update={"outcome": outcome, "decline_stage": stage}))
    with (out / RECORDS_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(r.model_dump_json() + "\n" for r in lines)

    return out, write_reports(out)
