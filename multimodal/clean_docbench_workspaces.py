from __future__ import annotations

import argparse
from pathlib import Path


DERIVED_NAMES = {
    "all_entities.json",
    "community.json",
    "entity.jsonl",
    "entity_media.json",
    "generate_relations.json",
    "graph_build_error.json",
    "leanrag_chunk.json",
    "manifest.json",
    "mm_chunk.json",
    "mm_media.json",
    "relation.jsonl",
}

DERIVED_SUFFIXES = {
    ".db",
    ".db-shm",
    ".db-wal",
    ".sqlite",
    ".sqlite3",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite3-shm",
    ".sqlite3-wal",
}


def clean_workspaces(working_root: str, apply: bool = False) -> list[Path]:
    root = Path(working_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"working_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"working_root is not a directory: {root}")

    targets = []
    for doc_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (doc_dir / "mineru_output").is_dir():
            continue
        for child in sorted(doc_dir.iterdir()):
            if child.name == "mineru_output":
                continue
            if child.is_file() and _is_derived_file(child):
                targets.append(child)

    for target in targets:
        action = "DELETE" if apply else "DRY-RUN"
        print(f"{action}\t{target}")
        if apply:
            target.unlink()
    print(f"{'Deleted' if apply else 'Would delete'} {len(targets)} files under {root}")
    return targets


def _is_derived_file(path: Path) -> bool:
    if path.name in DERIVED_NAMES:
        return True
    return any(path.name.endswith(suffix) for suffix in DERIVED_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean DocBench/MMLongBench document workspaces so each document directory keeps "
            "mineru_output/ while derived JSON/JSONL/DB artifacts are removed."
        )
    )
    parser.add_argument("working_root", help="Root directory containing one subdirectory per document.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Without this, only prints a dry run.")
    args = parser.parse_args()
    clean_workspaces(args.working_root, apply=args.apply)


if __name__ == "__main__":
    main()
