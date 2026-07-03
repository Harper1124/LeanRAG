from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


# 调用 MinerU 解析 PDF，并返回后续 chunk 构建所需的关键文件位置。
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
        # force=True 时清掉旧解析产物，保证重新解析。
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if not _find_content_list(out):
        # 如果已经有 content_list，则复用旧结果；否则调用 MinerU CLI。
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
    # 兼容不同 MinerU 版本的命令名称和 backend 参数名。
    commands = [
        ["magic-pdf", "-p", str(pdf), "-o", str(out), "-m", mineru_backend],
        ["mineru", "-p", str(pdf), "-o", str(out), "-b", mineru_backend],
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
    # content_list 优先，其它 JSON 作为兜底；middle.json 通常不是最终内容列表。
    candidates = sorted(out.rglob("*content_list*.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(path for path in out.rglob("*.json") if "middle" not in path.name.lower())
    return candidates[0] if candidates else None


def _find_markdown(out: Path) -> Path | None:
    candidates = sorted(out.rglob("*.md"))
    return candidates[0] if candidates else None


def _first_existing_dir(root: Path, names: list[str]) -> Path:
    # 不同解析版本图片/表格目录名不一致，这里按常见名称尝试匹配。
    for name in names:
        matches = [path for path in root.rglob(name) if path.is_dir()]
        if matches:
            return matches[0]
    return root
