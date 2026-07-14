from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from src.math.math_normalize import normalize_for_latex

IMAGE_LINE_RE = re.compile(r"^!\[\]\(([^)]+)\)\s*$")
MATH_PART_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$]+?\$)")


def normalize_math_markdown(text: str) -> str:
    return normalize_for_latex(text)


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "`": r"\textasciigrave{}",
    }
    out = text
    for char, repl in replacements.items():
        out = out.replace(char, repl)
    return out


def _format_prose_markdown(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\textit{\1}", text)
    return text


def _escape_outside_math(line: str) -> str:
    parts = MATH_PART_RE.split(line)
    out: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            # amsmath rejects tagged expressions inside inline math. Promote
            # them to display math so imported numbered equations remain
            # valid LaTeX instead of breaking a whole PDF export.
            if r"\tag" in part:
                if part.startswith("$$") and part.endswith("$$"):
                    part = f"\\[{part[2:-2]}\\]"
                elif part.startswith("$") and part.endswith("$"):
                    part = f"\\[{part[1:-1]}\\]"
            out.append(part)
        else:
            escaped = _escape_latex(part)
            out.append(_format_prose_markdown(escaped))
    return "".join(out)


def _merge_broken_math_lines(lines: list[str]) -> list[str]:
    """Join at most one continuation line when inline `$` delimiters are unbalanced."""
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.count("$") % 2 == 1 and i + 1 < len(lines):
            line = f"{line} {lines[i + 1].strip()}"
            i += 1
        merged.append(line)
        i += 1
    return merged


def _markdown_body(markdown: str, *, resource_dir: Path) -> list[str]:
    markdown = normalize_for_latex(markdown)
    body: list[str] = []

    for raw_line in _merge_broken_math_lines(markdown.splitlines()):
        stripped = raw_line.strip()
        if not stripped:
            body.append("")
            continue
        if stripped == "---":
            body.append("\\par\\medskip\\hrule\\medskip")
            continue
        if stripped.startswith("# "):
            body.append(f"\\section*{{{_escape_latex(stripped[2:])}}}")
            continue
        if stripped.startswith("## "):
            body.append(f"\\subsection*{{{_escape_latex(stripped[3:])}}}")
            continue
        if stripped.startswith("_") and stripped.endswith("_"):
            body.append(f"\\textit{{{_escape_outside_math(stripped.strip('_'))}}}")
            continue

        image_match = IMAGE_LINE_RE.match(stripped)
        if image_match:
            ref = image_match.group(1).replace("\\", "/")
            img_path = Path(ref)
            if not img_path.is_absolute():
                img_path = resource_dir / ref
            if img_path.is_file():
                body.append(
                    "\\begin{center}"
                    f"\\includegraphics[width=0.75\\linewidth]{{{img_path.as_posix()}}}"
                    "\\end{center}"
                )
            continue

        body.append(_escape_outside_math(raw_line.rstrip()))

    return body


def markdown_to_latex_document(markdown: str, *, resource_dir: Path, engine: str) -> str:
    body = _markdown_body(markdown, resource_dir=resource_dir)
    body_text = "\n".join(body)

    if engine == "xelatex":
        preamble = (
            "\\documentclass[11pt,a4paper]{article}\n"
            "\\usepackage[margin=2cm]{geometry}\n"
            "\\usepackage{fontspec}\n"
            "\\usepackage{amsmath,amssymb}\n"
            "\\usepackage{graphicx}\n"
            "\\setmainfont{DejaVu Serif}\n"
        )
    else:
        preamble = (
            "\\documentclass[11pt,a4paper]{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{lmodern}\n"
            "\\usepackage[margin=2cm]{geometry}\n"
            "\\usepackage{amsmath,amssymb}\n"
            "\\usepackage{graphicx}\n"
        )

    return preamble + "\\begin{document}\n" + body_text + "\n\\end{document}\n"


def export_pdf_via_latex(
    markdown_path: Path,
    pdf_path: Path,
    *,
    resource_dir: Path,
) -> tuple[bool, str]:
    markdown = markdown_path.read_text(encoding="utf-8")
    engines = [name for name in ("xelatex", "pdflatex") if shutil.which(name)]
    if not engines:
        return False, "No LaTeX engine found (install texlive-xetex or texlive-latex-base)"

    last_error = ""
    for engine in engines:
        tex_path = pdf_path.with_suffix(".tex")
        tex_path.write_text(
            markdown_to_latex_document(markdown, resource_dir=resource_dir, engine=engine),
            encoding="utf-8",
        )
        for _ in range(2):
            proc = subprocess.run(
                [
                    engine,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(pdf_path.parent),
                    tex_path.name,
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(pdf_path.parent),
            )
            last_error = proc.stderr or proc.stdout or last_error
            if proc.returncode != 0:
                break
        if pdf_path.is_file():
            return True, engine

    log_path = pdf_path.with_suffix(".log")
    if log_path.is_file():
        last_error = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
    return False, last_error or "LaTeX PDF export failed"


def export_pdf(markdown_path: Path, pdf_path: Path, *, resource_dir: Path) -> tuple[bool, str]:
    pandoc = shutil.which("pandoc")
    if pandoc:
        for engine in ("xelatex", "pdflatex"):
            if not shutil.which(engine):
                continue
            proc = subprocess.run(
                [
                    pandoc,
                    str(markdown_path),
                    "-o",
                    str(pdf_path),
                    f"--resource-path={resource_dir}",
                    f"--pdf-engine={engine}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and pdf_path.is_file():
                return True, f"pandoc/{engine}"

    return export_pdf_via_latex(
        markdown_path,
        pdf_path,
        resource_dir=resource_dir,
    )
