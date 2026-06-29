from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def parse_pdf_with_mineru(
    pdf_path: str,
    output_dir: str,
    mineru_backend: str = "pipeline",
    force: bool = False,
) -> dict:
    """
    Run MinerU and return important output paths.

    If output_dir already contains MinerU artifacts, they are reused unless force=True.
    MinerU CLIs differ by version; this function first tries `magic-pdf`, then `mineru`.
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    out = Path(output_dir)
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if not _find_content_list(out):
        _run_mineru_cli(pdf, out, mineru_backend)

    content_list = _find_content_list(out)
    markdown = _find_markdown(out)
    return {
        "pdf_path": str(pdf),
        "mineru_output_dir": str(out),
        "markdown_file": str(markdown) if markdown else None,
        "content_list_file": str(content_list) if content_list else None,
        "json_files": [str(path) for path in sorted(out.rglob("*.json"))],
        "image_dir": str(_first_existing_dir(out, ["images", "image", "imgs"])),
        "table_dir": str(_first_existing_dir(out, ["tables", "table"])),
        "page_info": {},
    }


def _run_mineru_cli(pdf: Path, out: Path, mineru_backend: str) -> None:
    commands = [
        ["magic-pdf", "-p", str(pdf), "-o", str(out), "-m", mineru_backend],
        ["mineru", "-p", str(pdf), "-o", str(out), "-m", mineru_backend],
    ]
    errors = []
    for command in commands:
        try:
            subprocess.run(command, check=True)
            return
        except FileNotFoundError as exc:
            errors.append(str(exc))
        except subprocess.CalledProcessError as exc:
            errors.append(f"{' '.join(command)} exited with {exc.returncode}")
    raise RuntimeError(
        "MinerU CLI failed or was not found. Install MinerU and ensure magic-pdf/mineru is on PATH. "
        + " | ".join(errors)
    )


def _find_content_list(out: Path) -> Path | None:
    candidates = sorted(out.rglob("*content_list*.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(path for path in out.rglob("*.json") if "middle" not in path.name.lower())
    return candidates[0] if candidates else None


def _find_markdown(out: Path) -> Path | None:
    candidates = sorted(out.rglob("*.md"))
    return candidates[0] if candidates else None


def _first_existing_dir(root: Path, names: list[str]) -> Path:
    for name in names:
        matches = [path for path in root.rglob(name) if path.is_dir()]
        if matches:
            return matches[0]
    return root
