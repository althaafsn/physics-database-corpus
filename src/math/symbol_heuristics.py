"""Deterministic fixes for common OCR blank-symbol patterns in olympiad problems."""
from __future__ import annotations

import re

# Coupled oscillators / spring-mass (OSN 2019 #1 style)
_COUPLED_MASS_SPRINGS = re.compile(
    r"massa\s*,\s*konstanta pegas dan\s*,",
    re.IGNORECASE,
)
_COUPLED_AMP_LIST = re.compile(
    r"dinyatakan dalam 0,\s*,\s*,\s*dan\s*\)",
    re.IGNORECASE,
)
_F0_COS = re.compile(r"\(\s*\)\s*=\s*\$?\^?0\$?\s*cos", re.IGNORECASE)
_F_EXTERNAL_EMPTY = re.compile(r"gaya eksternal \(\s*\)", re.IGNORECASE)
_BLANK_AND_BANG = re.compile(
    r"(?:\$T_\{\\rm env\}\$|T_\{\\rm env\}|α),\s*dan\s*!",
)
_BLANK_VOLTAGE_LIST = re.compile(
    r"\$T_\{\\rm env\}\$,\s*dan\s*!",
)
_INERTIA_MR2 = re.compile(
    r"adalah\s*=\s*2\s*/2\s+dan terhadap sumbu putar yang sejajar",
    re.IGNORECASE,
)
_INERTIA_MR4 = re.compile(
    r"adalah\s*=\s*2\s*/4\.",
    re.IGNORECASE,
)


def apply_symbol_heuristics(text: str) -> str:
    text = _COUPLED_MASS_SPRINGS.sub("massa $m$, konstanta pegas $k_a$ dan $k_b$,", text)
    text = _COUPLED_AMP_LIST.sub(
        lambda _m: "dinyatakan dalam $F_0$, $\\gamma$, $k_a$, dan $k_b$)",
        text,
    )
    text = _F0_COS.sub(lambda _m: "$F(t)$ = $F_0$ cos", text)
    text = _F_EXTERNAL_EMPTY.sub("gaya eksternal $F(t)$", text)
    text = _BLANK_VOLTAGE_LIST.sub("$T_{\\rm env}$, dan $V$!", text)
    text = _BLANK_AND_BANG.sub("$T_{\\rm env}$, dan $V$!", text)
    text = re.sub(
        r"dan\s*!\s*Cari juga nilai numerik",
        "dan $V$! Cari juga nilai numerik",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"dalam \$h\$,\s*\$A\$,\s*\$T_c\$,\s*α,\s*dan\s*!",
        "dalam $h$, $A$, $T_c$, $α$, dan $V$!",
        text,
        flags=re.IGNORECASE,
    )
    text = _INERTIA_MR2.sub(
        "adalah $I = mr^2/2$ dan terhadap sumbu putar yang sejajar",
        text,
    )
    text = _INERTIA_MR4.sub("adalah $I = mr^2/4$.", text)
    text = re.sub(
        r"bermassa\s+dan berjari-jari\s+dan memiliki",
        "bermassa $m$ dan berjari-jari $r$ dan memiliki",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"panjang\s+\(massa batang",
        "panjang $L$ (massa batang",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Besar percepatan gravitasi adalah\s*\.",
        "Besar percepatan gravitasi adalah $g$.",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Tentukan dinyatakan dalam\s*,\s*,\s*,\s*dan\s*!",
        lambda _m: "Tentukan dinyatakan dalam $\\dot{R}/R$, $\\omega$, $M_B$, $M_M$, $R_B$, $G$, dan $R$!",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\(dinyatakan dalam\s*,\s*,\s*,\s*1,\s*dan\s*2\)",
        lambda _m: "(dinyatakan dalam $\\rho$, $A$, $S$, $v_1$, dan $v_2$)",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"dinyatakan dalam\s*,\s*,\s*,\s*1,\s*dan\s*2\)",
        lambda _m: "dinyatakan dalam $\\rho$, $A$, $S$, $v_1$, dan $v_2$)",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"dinyatakan dalam\s*,\s*,\s*,\s*dan\s*dimana\s*2",
        "dinyatakan dalam $m$, $v_0$, $R$, $G$, dan $a$ dimana $a$",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"berjari-jari\s+dan\s+bermassa\s*,",
        "berjari-jari $R$ dan bermassa $M$,",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"variabel-variabel\s+0,\s*ℎ,\s*,\s*,\s*,\s*dan\s*\$V\$!",
        "variabel-variabel $R_0$, $h$, $A$, $T_c$, $α$, dan $V$!",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"variabel-variabel\s+\$R_0\$,\s*\$h\$,\s*\$A\$,\s*\$T_c\$,\s*α,\s*dan\s*!",
        "variabel-variabel $R_0$, $h$, $A$, $T_c$, $α$, dan $V$!",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"sebagai bola dengan jari-jari dan bermassa\s*,",
        "sebagai bola dengan jari-jari $R$ dan bermassa $M$,",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\(dinyatakan dalam\s*,\s*,\s*,\s*,\s*,\s*,\s*dan\s*\)",
        lambda _m: "(dinyatakan dalam $M_B$, $M_M$, $R_B$, $R_0$, $m$, $G$, dan $\\tau$)",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"dan nyatakan dalam\s*,\s*,\s*dan\s*!",
        lambda _m: "dan nyatakan dalam $h$, $A$, $T_c$, $α$, dan $V$!",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"nyatakan dalam \$R\^0\$,\s*,\s*,\s*,\s*,\s*dan\s*\\\$\\omega\^0\$",
        lambda _m: "nyatakan dalam $R_0$, $M_B$, $M_M$, $R_B$, $G$, dan $\\omega_0$",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"nyatakan dalam \$R\^0\$,\s*,\s*,\s*,\s*,\s*dan\s*\$\\omega\^0\$",
        lambda _m: "nyatakan dalam $R_0$, $M_B$, $M_M$, $R_B$, $G$, dan $\\omega_0$",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Anda dalam\s*,\s*ℎ,\s*,\s*,\s*,\s*dimana",
        lambda _m: "Anda dalam $A$, $h$, $T_c$, $α$, $T_{\\rm ruangan}$, dan $T_{\\rm mendidih}$, dimana",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"jawaban Anda dalam 0,\s*ℎ,\s*,\s*,\s*dan\s*\.",
        lambda _m: "jawaban Anda dalam $R_0$, $h$, $A$, $T_c$, $α$, dan $I$.",
        text,
        flags=re.IGNORECASE,
    )
    return text
