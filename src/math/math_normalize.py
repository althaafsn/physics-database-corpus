"""Normalize physics markdown for pdflatex / PDF export."""

from __future__ import annotations

import re
import unicodedata

MATH_PART_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$]+?\$)")
MATH_ITALIC_CHAR_RE = re.compile(r"[\U0001D400-\U0001D7FF]")

_GREEK_ITALIC_SMALL = "αβγδεζηθικλμνξοπρστυφχψω"
_GREEK_ITALIC_LATEX = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
)
_GREEK_ITALIC_CAP_LATEX = (
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zeta",
    "Eta",
    "Theta",
    "Iota",
    "Kappa",
    "Lambda",
    "Mu",
    "Nu",
    "Xi",
    "Omicron",
    "Pi",
    "Rho",
    "Sigma",
    "Tau",
    "Upsilon",
    "Phi",
    "Chi",
    "Psi",
    "Omega",
)

# Unicode → LaTeX command (used inside math; prose wraps with $...$).
MATH_UNICODE: tuple[tuple[str, str], ...] = (
    ("α", r"\alpha"),
    ("β", r"\beta"),
    ("γ", r"\gamma"),
    ("δ", r"\delta"),
    ("ε", r"\epsilon"),
    ("η", r"\eta"),
    ("θ", r"\theta"),
    ("μ", r"\mu"),
    ("µ", r"\mu"),
    ("ν", r"\nu"),
    ("ξ", r"\xi"),
    ("π", r"\pi"),
    ("ρ", r"\rho"),
    ("φ", r"\phi"),
    ("ω", r"\omega"),
    ("ū", r"\hat{u}"),
    ("Ū", r"\hat{U}"),
    ("ȧ", r"\dot{a}"),
    ("λ", r"\lambda"),
    ("∆", r"\Delta"),
    ("⊙", r"\odot"),
    ("⋅", r"\cdot"),
    ("∫", r"\int"),
    ("∑", r"\sum"),
    ("∠", r"\angle"),
    ("≃", r"\simeq"),
    ("∝", r"\propto"),
    ("⟨", r"\langle"),
    ("⟩", r"\rangle"),
    ("‖", r"\|"),
    ("ħ", r"\hbar"),
    ("ℏ", r"\hbar"),
    ("ℳ", r"\mathcal{M}"),
    ("•", r"\bullet"),
    ("Δ", r"\Delta"),
    ("Φ", r"\Phi"),
    ("Ω", r"\Omega"),
    ("ℎ", "h"),
    ("ℓ", r"\ell"),
    ("°", r"^{\circ}"),
    ("∘", r"^{\circ}"),
    ("±", r"\pm"),
    ("×", r"\times"),
    ("·", r"\cdot"),
    ("≈", r"\approx"),
    ("≠", r"\neq"),
    ("≤", r"\leq"),
    ("≥", r"\geq"),
    ("≪", r"\ll"),
    ("≫", r"\gg"),
    ("⋯", r"\cdots"),
    ("∞", r"\infty"),
    ("→", r"\rightarrow"),
    ("−", "-"),
    ("²", "^{2}"),
    ("³", "^{3}"),
    ("₁", "_1"),
    ("₂", "_2"),
    ("₃", "_3"),
    ("ᵢ", "_i"),
    ("ₖ", "_k"),
    ("ₘ", "_m"),
    ("ₙ", "_n"),
    ("ₚ", "_p"),
    ("ₛ", "_s"),
    ("ₐ", "_a"),
    ("…", r"\ldots"),
    ("√", r"\sqrt{}"),
    ("′", "'"),
    ("–", "-"),
    ("—", "-"),
    ("⃗", r"\vec{}"),
)


def fix_literal_unicode_escapes(text: str) -> str:
    """Decode literal \\u0394 sequences stored as text instead of real Unicode."""
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )

TAB_COMMAND_FIXES: tuple[tuple[str, str], ...] = (
    ("heta", r"\theta"),
    ("extit{", r"\textit{"),
    ("ext{", r"\text{"),
    ("au", r"\tau"),
    ("imes", r"\times"),
    ("ilde{", r"\tilde{"),
    ("o ", r"\to "),
    ("an", r"\tan"),
    ("an(", r"\tan("),
)

GREEK_IN_TEXT = re.compile(
    r"\\text\{(\\(?:alpha|beta|gamma|delta|epsilon|eta|theta|mu|nu|xi|pi|rho|phi|omega|Delta|Phi|Omega))\}"
)
TEXT_CMD_RE = re.compile(r"\\text\{([^}]*)\}")


def _normalize_text_commands(segment: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = _map_unicode_to_latex(match.group(1)).strip()
        if re.fullmatch(r"(\\[A-Za-z]+\s*)+", body):
            return body
        if re.fullmatch(r"\\[A-Za-z]+", body):
            return body
        return f"\\text{{{body}}}"

    return TEXT_CMD_RE.sub(repl, segment)


def fix_degree_markers(text: str) -> str:
    text = re.sub(r"\$(\d+)\^°\$", r"$\1^{\\circ}$", text)
    text = text.replace("^°", "^{\\circ}")
    return text


def fix_ll_corruption(text: str) -> str:
    """Repair `$\\ll$` mangled as `$` + newline + `n` (JSON `\\n` swallowing a backslash)."""
    text = text.replace("\\nn", "\\ll")
    text = re.sub(r"(\$[^\n$]*)\n\s*n(\s)", r"\1 \\ll \2", text)
    return text


def fix_rho_corruption(text: str) -> str:
    """Repair $\\rho$ stored as `$` + CR/LF + `ho` (JSON `\\rho` → Python `\\r` + `ho`)."""
    text = (
        text.replace("\x0crho", "\\rho")
        .replace("\r" + "ho", "\\rho")
    )
    # read_text() may convert lone CR to LF before we see it
    text = re.sub(r"\$[\r\n]\s*ho([^$]*?\$)", r"$\\rho\1", text)
    text = re.sub(r"\$\s+ho(?=[,$\s)])", r"$\\rho$", text)
    return text


def fix_json_control_artifacts(text: str) -> str:
    text = (
        text.replace("\x08oldsymbol{", "\\boldsymbol{")
        .replace("\x09ext{", "\\text{")
        .replace("\x0crac{", "\\frac{")
        .replace("\u2009", " ")
        .replace("\u00a0", " ")
    )
    text = fix_rho_corruption(text)
    text = fix_ll_corruption(text)
    text = re.sub(r"\x08(?=[a-zA-Z])", r"\\b", text)
    for suffix, cmd in TAB_COMMAND_FIXES:
        text = text.replace("\t" + suffix, cmd)
    text = re.sub(r"(?<!\$)\$rac\{", r"$\\frac{", text)
    # Generic fallback: any remaining \f or \r control char immediately
    # followed by letters was almost certainly a LaTeX command (\frac,
    # \rho, \right...) swallowed by json.loads treating \f/\r as legal
    # JSON control escapes. \x09 (tab) and \x08 (backspace) are handled
    # above/via TAB_COMMAND_FIXES already.
    text = re.sub(r"\x0c(?=[a-zA-Z])", r"\\f", text)
    text = re.sub(r"\x0d(?=[a-zA-Z])", r"\\r", text)
    return text


def fix_html_markup(text: str) -> str:
    text = re.sub(
        r"<sup>([^<]+)</sup>",
        lambda m: f"$^{{{_strip_html_inner(m.group(1))}}}$",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<sub>([^<]+)</sub>",
        lambda m: f"$_{{{_strip_html_inner(m.group(1))}}}$",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _strip_html_inner(value: str) -> str:
    return value.strip()


def is_math_italic_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return 0x1D400 <= code <= 0x1D7FF


def math_italic_char_to_latex(ch: str) -> str:
    """Map one Mathematical Alphanumeric Symbol to a LaTeX math fragment."""
    code = ord(ch)
    if 0x1D434 <= code <= 0x1D44D:
        return chr(code - 0x1D434 + ord("A"))
    if 0x1D44E <= code <= 0x1D467:
        return chr(code - 0x1D44E + ord("a"))
    if 0x1D6FC <= code <= 0x1D717:
        idx = code - 0x1D6FC
        if idx < len(_GREEK_ITALIC_LATEX):
            return f"\\{_GREEK_ITALIC_LATEX[idx]}"
    if 0x1D6E2 <= code <= 0x1D6FA:
        idx = code - 0x1D6E2
        if idx < len(_GREEK_ITALIC_CAP_LATEX):
            name = _GREEK_ITALIC_CAP_LATEX[idx]
            if name in {"Delta", "Phi", "Omega", "Pi", "Sigma", "Theta"}:
                return f"\\{name}"
            return name[0]
    # Bold/digit variants and other math alphanumeric blocks — fall back to ASCII lookalike.
    if 0x1D400 <= code <= 0x1D7FF:
        normalized = unicodedata.normalize("NFKD", ch)
        ascii_part = "".join(c for c in normalized if c.isascii() and c.isalnum())
        if ascii_part:
            return ascii_part
    return ch


_GREEK_LATEX_CMDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zeta",
    "Eta",
    "Theta",
    "Iota",
    "Kappa",
    "Lambda",
    "Mu",
    "Nu",
    "Xi",
    "Omicron",
    "Pi",
    "Rho",
    "Sigma",
    "Tau",
    "Upsilon",
    "Phi",
    "Chi",
    "Psi",
    "Omega",
)
_GREEK_LATEX_CMD_RE = "|".join(_GREEK_LATEX_CMDS)


def _math_italic_run_to_latex(segment: str, start: int) -> tuple[str, int]:
    """Consume a math-italic letter plus trailing primes / combining marks."""
    ch = segment[start]
    base = math_italic_char_to_latex(ch)
    index = start + 1
    combining: list[str] = []
    while index < len(segment):
        nxt = segment[index]
        if nxt == "\u2032":
            combining.append("prime")
            index += 1
            continue
        if unicodedata.category(nxt) == "Mn":
            combining.append(nxt)
            index += 1
            continue
        break

    sub_digits = ""
    while index < len(segment) and segment[index].isascii() and segment[index].isdigit():
        sub_digits += segment[index]
        index += 1

    if "\u0307" in combining:
        latex = f"\\dot{{{base}}}"
    elif "\u0302" in combining:
        latex = f"\\hat{{{base}}}"
    elif "prime" in combining:
        latex = f"{base}'"
    elif base.startswith("\\"):
        latex = base
    else:
        latex = base

    if sub_digits:
        latex = f"{latex}_{{{sub_digits}}}"
    return f"${latex}$", index


def convert_math_italic_runs(segment: str) -> str:
    """Wrap PDF Mathematical Italic symbols (𝑉̇, 𝑃′, 𝜂) in $...$ / \\dot{} LaTeX."""
    segment = unicodedata.normalize("NFC", segment)
    out: list[str] = []
    index = 0
    while index < len(segment):
        ch = segment[index]
        if is_math_italic_char(ch):
            latex, index = _math_italic_run_to_latex(segment, index)
            out.append(latex)
            continue
        if ch == "\u2032" and out and out[-1].endswith("$") and not out[-1].endswith("'$"):
            out[-1] = out[-1][:-1] + "'$"
            index += 1
            continue
        out.append(ch)
        index += 1
    return "".join(out)


def _map_unicode_to_latex(value: str) -> str:
    for uni, latex in MATH_UNICODE:
        value = value.replace(uni, latex)
    return value


def _fix_latex_math_commands(segment: str) -> str:
    segment = re.sub(
        r"\\(alpha|beta|gamma|delta|epsilon|eta|theta|mu|nu|xi|pi|rho|phi|omega|Delta|Phi|Omega)(?=[A-Za-z])",
        r"\\\1 ",
        segment,
    )
    segment = re.sub(r"\\text\{sin\}", r"\\sin", segment, flags=re.IGNORECASE)
    segment = re.sub(r"\\text\{cos\}", r"\\cos", segment, flags=re.IGNORECASE)
    segment = re.sub(r"\\text\{tan\}", r"\\tan", segment, flags=re.IGNORECASE)
    return segment


def _fix_dollar_caret_block(match: re.Match[str], text: str) -> str:
    inner = _map_unicode_to_latex(match.group(1).strip())
    before = text[: match.start()].rstrip()

    if inner in {"0", r"\circ"} and before and before[-1].isdigit():
        return r"$^{\circ}$"

    if inner.isdigit():
        unit_tail = before[-1:] if before else ""
        if unit_tail in {"/", "m", "s", "g", "k", "N", "J", "W", "H", "V", "A", "L"}:
            return f"$^{{{inner}}}$"
        if len(before) >= 2 and before[-2:] in {"/s", "m/", "g/"}:
            return f"$^{{{inner}}}$"

    if len(inner) <= 3 and not inner.startswith("{"):
        return f"$_{{{inner}}}$"

    return f"$^{{{inner}}}$"


def fix_dollar_caret_math(text: str) -> str:
    return re.sub(
        r"\$\^(?!\{)([^$]+)\$",
        lambda m: _fix_dollar_caret_block(m, text),
        text,
    )


def fix_unbraced_inline_exponents(text: str) -> str:
    return re.sub(r"\$\^([^{][^$]*)\$", r"$^{\1}$", text)


def fix_empty_display_math(text: str) -> str:
    return re.sub(r"\$\$\s*\$\$", "", text)


def _normalize_math_segment(segment: str) -> str:
    # Strip inline delimiters; display $$...$$ is handled as a single segment.
    display = segment.startswith("$$") and segment.endswith("$$")
    if display:
        inner = segment[2:-2]
    elif segment.startswith("$") and segment.endswith("$"):
        inner = segment[1:-1]
    else:
        inner = segment

    inner = _map_unicode_to_latex(inner)
    inner = "".join(
        math_italic_char_to_latex(c) if is_math_italic_char(c) else c for c in inner
    )
    inner = _fix_latex_math_commands(inner)
    inner = _normalize_text_commands(inner)
    inner = GREEK_IN_TEXT.sub(r"\1", inner)
    inner = re.sub(r"(?<![\\$])\^(?!\{)([A-Za-z0-9])", r"^{\1}", inner)
    inner = re.sub(r"(?<!\\)_([A-Za-z0-9])", r"_{\1}", inner)
    inner = re.sub(rf"\\({_GREEK_LATEX_CMD_RE})\s+(\d+)", r"\\\1_{\2}", inner)

    if display:
        return f"$${inner}$$"
    if segment.startswith("$") and segment.endswith("$"):
        return f"${inner}$"
    return inner


def _normalize_prose_segment(segment: str) -> str:
    segment = unicodedata.normalize("NFC", segment)
    segment = convert_math_italic_runs(segment)
    segment = re.sub(r"([A-Za-z])⃗", r"$\\vec{\1}$", segment)
    segment = re.sub(r"([A-Za-z])\u0302", r"$\\hat{\1}$", segment)
    segment = re.sub(r"([A-Za-z])\u0307", r"$\\dot{\1}$", segment)
    for uni, latex in MATH_UNICODE:
        segment = segment.replace(uni, f"${latex}$")
    segment = "".join(c for c in segment if unicodedata.category(c) != "Mn")
    return segment


def _split_and_map(text: str) -> str:
    parts = MATH_PART_RE.split(text)
    out: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            out.append(_normalize_math_segment(part))
        else:
            out.append(_normalize_prose_segment(part))
    return "".join(out)


def fix_split_subscript_digits(text: str) -> str:
    """Repair $\\alpha$2, $\\alpha$ 2, and \\alpha 2 → $\\alpha_2$ (and $v$0 → $v_0$)."""
    greek = _GREEK_LATEX_CMD_RE

    def _greek_sub(match: re.Match[str]) -> str:
        return f"$\\{match.group(1)}_{{{match.group(2)}}}$"

    text = re.sub(rf"\$\\({greek})\$(\d+)", _greek_sub, text)
    text = re.sub(rf"\$\\({greek})\$\s+(\d+)", _greek_sub, text)
    text = re.sub(rf"(?<!\$)\\({greek})\s+(\d+)(?!\$)", _greek_sub, text)
    text = re.sub(r"\$([A-Za-z])\$(\d+)", r"$\1_{\2}$", text)
    text = re.sub(r"\$([A-Za-z])\$\s+(\d+)", r"$\1_{\2}$", text)

    def _fix_math_inner(match: re.Match[str]) -> str:
        inner = match.group(1)
        inner = re.sub(rf"\\({greek})\s+(\d+)", r"\\\1_{\2}", inner)
        return f"${inner}$"

    return re.sub(r"\$([^$]+)\$", _fix_math_inner, text)


def fix_unclosed_inline_math(text: str) -> str:
    """Close `$v_2.` style math where the trailing `$` was lost."""
    return re.sub(
        r"(?<![A-Za-z0-9\\])\$([^$\n]{0,40}?[_^][A-Za-z0-9]+)([.!?;:])(?!\$)",
        r"$\1$\2",
        text,
    )


def strip_combining_marks(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


_RESIDUAL_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_residual_control_chars(text: str) -> str:
    """Remove C0 control chars that survive targeted repair (OCR/json artifacts).

    Keeps tab/newline/carriage-return, which are meaningful whitespace.
    """
    return _RESIDUAL_CTRL_RE.sub("", text)


# Typographic / smart punctuation → plain ASCII (avoids wrapping in math $...$).
TYPOGRAPHIC_ASCII: tuple[tuple[str, str], ...] = (
    ("\u201c", '"'),   # left double quote
    ("\u201d", '"'),   # right double quote
    ("\u2018", "'"),   # left single quote
    ("\u2019", "'"),   # right single quote
    ("\u2013", "-"),   # en dash
    ("\u2014", "-"),   # em dash
    ("\u00a0", " "),   # no-break space
    ("\u2009", " "),   # thin space
    ("\u202f", " "),   # narrow no-break space
    ("\u2044", "/"),   # fraction slash
    ("\u00a9", "(c)"), # copyright
    ("\u00ad", ""),    # soft hyphen
)


def fix_typographic_punctuation(text: str) -> str:
    """Replace smart quotes / dashes / spaces with ASCII before math splitting."""
    for uni, repl in TYPOGRAPHIC_ASCII:
        text = text.replace(uni, repl)
    return text


# Symbols with no LaTeX form — PDF-extraction artifacts (font PUA glyphs,
# bracket pieces, parenthesized letters). Dropped as a last resort.
_NONRENDERABLE_CHARS = frozenset(
    "\u24a7\u24ad"          # ⒧ ⒭ parenthesized small letters
    "\u23a1\u23a3\u23a4\u23a6"  # ⎡ ⎣ ⎤ ⎦ bracket pieces
)


def _final_sanitize(text: str) -> str:
    """Fold compatibility forms (ligatures, accents) and drop unrenderable residue."""
    text = unicodedata.normalize("NFKD", text)
    # NFKD can fold math-italic glyphs (e.g. 𝜔) into plain Greek letters that
    # the earlier per-segment map never saw; re-map before stripping.
    text = _map_unicode_to_latex(text)
    out: list[str] = []
    for ch in text:
        if unicodedata.category(ch) in {"Mn", "Cf"}:
            continue
        code = ord(ch)
        if 0xE000 <= code <= 0xF8FF:  # Private Use Area
            continue
        if ch in _NONRENDERABLE_CHARS:
            continue
        out.append(ch)
    return "".join(out)


def normalize_problem_symbols(text: str) -> str:
    """Deterministic Unicode math → inline LaTeX for silver/gold problem bodies."""
    if not text or not text.strip():
        return text
    text = fix_json_control_artifacts(text)
    text = fix_html_markup(text)
    parts = MATH_PART_RE.split(text)
    out: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            out.append(part)
        else:
            segment = unicodedata.normalize("NFC", part)
            segment = convert_math_italic_runs(segment)
            for uni, latex in MATH_UNICODE:
                if uni in ("′", "–", "—"):
                    continue
                segment = segment.replace(uni, f"${latex}$")
            segment = re.sub(r"([A-Za-z])⃗", r"$\\vec{\1}$", segment)
            segment = re.sub(r"([A-Za-z])\u0302", r"$\\hat{\1}$", segment)
            segment = re.sub(r"([A-Za-z])\u0307", r"$\\dot{\1}$", segment)
            segment = "".join(c for c in segment if unicodedata.category(c) != "Mn")
            out.append(segment)
    text = "".join(out)
    return fix_split_subscript_digits(text)


def normalize_for_latex(text: str) -> str:
    """Apply all markdown → LaTeX-safe normalization passes."""
    text = fix_json_control_artifacts(text)
    text = fix_typographic_punctuation(text)
    text = fix_literal_unicode_escapes(text)
    text = fix_degree_markers(text)
    text = fix_html_markup(text)
    text = fix_empty_display_math(text)
    text = fix_dollar_caret_math(text)
    text = fix_unclosed_inline_math(text)
    text = _split_and_map(text)
    text = fix_unbraced_inline_exponents(text)
    text = strip_combining_marks(text)
    text = strip_residual_control_chars(text)
    return _final_sanitize(text)
