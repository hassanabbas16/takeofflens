"""Finnish room labels -> English room types.

A pure lookup table with no LLM involvement, because it is also used to *score* LLM output.
Frequencies in the comments are counts over 120 test-split model.svg files, so the common
cases are covered first.

Labels are matched case-insensitively after stripping punctuation and unescaping XML
entities (CubiCasa SVGs contain ``KEITTI&#xD6;`` for ``KEITTIÖ``). Unknown labels map to
``unknown`` with the raw text preserved - never guess a type.
"""

from __future__ import annotations

import re
import unicodedata
from html import unescape

UNKNOWN = "unknown"
UNDEFINED = "undefined"

# Finnish label -> English room_type. Grouped by what the label abbreviates.
ROOM_TYPES: dict[str, str] = {
    # Living / sleeping
    "MH": "bedroom",           # makuuhuone, 204
    "OH": "living_room",       # olohuone, 99
    "OLESKELU": "living_room",
    "H": "room",               # huone, 35
    "AH": "hobby_room",        # askarteluhuone, 10
    "ALKOVI": "alcove",
    # Kitchen / dining
    "K": "kitchen",            # keittiö, 73
    "KEITTIO": "kitchen",      # KEITTIÖ, 23
    "KK": "kitchen",           # keittokomero, 9
    "KEITTOKOMERO": "kitchen",
    "RUOK": "dining",          # ruokailu, 9
    "RUOKAILU": "dining",
    # Entry / circulation
    "ET": "entry",             # eteinen, 90
    "ETEINEN": "entry",
    "TK": "draught_lobby",     # tuulikaappi, 48
    "TUULIKAAPPI": "draught_lobby",
    "AULA": "hall",
    "KAYTAVA": "corridor",     # KÄYTÄVÄ
    "PORRAS": "stairs",
    # Wet rooms
    "WC": "wc",                # 94
    "KH": "bathroom",          # kylpyhuone, 28
    "KPH": "bathroom",         # 12
    "KYLPYHUONE": "bathroom",
    "PH": "washroom",          # pesuhuone, 47
    "PESUH": "washroom",       # 11
    "PSH": "washroom",         # 7
    "PESUHUONE": "washroom",
    "PKH": "washroom",
    "SH": "sauna",             # saunahuone, 7
    "SAUNA": "sauna",
    "S": "sauna",
    # Utility / storage
    "KHH": "utility",          # kodinhoitohuone, 33
    "VAR": "storage",          # varasto, 42
    "VARASTO": "storage",      # 12
    "TEKN": "technical_room",  # 8
    "TEKNINEN": "technical_room",
    "ULLAKKO": "attic",
    "APUK": "utility",         # apukeittiö
    # Closets / dressing
    "VH": "walk_in_closet",    # vaatehuone, 63
    "VAATEHUONE": "walk_in_closet",
    "PUKUH": "dressing_room",  # pukuhuone, 5
    "KOMERO": "closet",
    # Work
    "TYOHUONE": "office",      # TYÖHUONE
    "TH": "office",            # 7
    "RT": "office",
    # Outdoor
    "ULKOTILA": "outdoor",     # 131
    "PARVEKE": "balcony",      # 31
    "PARV": "balcony",
    "TERASSI": "terrace",      # 25
    "KUISTI": "porch",         # 17
    "AUTOKATOS": "carport",    # 9
    "AUTOTALLI": "garage",     # 7
    "KATTH": "covered_terrace",  # KATT.H
    # Annotation placeholder, not a real room type.
    "UNDEFINED": UNDEFINED,    # 243 - the single most common label in the dataset
}

# Longest first, so OLESKELU wins over a bare prefix when scanning compound labels.
_KNOWN_LABELS: tuple[str, ...] = tuple(sorted(ROOM_TYPES, key=len, reverse=True))


def normalise_label(text: str) -> str:
    """Upper-case, unescape entities, strip accents and punctuation.

    Accents are folded (Ä->A, Ö->O) rather than preserved because the OCR recogniser in use
    is the English model, which cannot emit them - so the lexicon has to meet it halfway.
    Revisit if a Finnish recogniser is adopted.
    """
    text = unescape(text).strip().upper()
    # NFKD then drop combining marks: Ä -> A, Ö -> O.
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9+]", "", text)


def lookup(text: str) -> str:
    """Return the English room type for a Finnish label, or ``unknown``."""
    return ROOM_TYPES.get(normalise_label(text), UNKNOWN)


def split_compound(text: str) -> list[str]:
    """Split a compound label such as ``OLESKELU+RUOK`` into its parts."""
    return [p for p in normalise_label(text).split("+") if p]


def lookup_compound(text: str) -> list[str]:
    """Room types for a compound label, in order. Unknown parts are kept as ``unknown``."""
    parts = split_compound(text)
    if not parts:
        return []
    return [ROOM_TYPES.get(p, UNKNOWN) for p in parts]


def primary_type(text: str) -> str:
    """The room type to store for a label; the first part of a compound."""
    types = lookup_compound(text)
    for candidate in types:
        if candidate != UNKNOWN:
            return candidate
    return types[0] if types else UNKNOWN


def is_known_label(text: str) -> bool:
    return lookup(text) != UNKNOWN


def label_match_score(text: str) -> float:
    """How strongly ``text`` looks like a Finnish room label, in [0, 1].

    Used to choose between competing OCR readings of the same box. Exact lexicon hits score
    highest; a compound whose parts are all known scores next; a reading that merely
    *contains* a known label scores low, so that ``TrOHUONE`` does not outrank ``TYOHUONE``.
    """
    normalised = normalise_label(text)
    if not normalised:
        return 0.0
    if normalised in ROOM_TYPES:
        return 1.0
    parts = split_compound(normalised)
    if len(parts) > 1 and all(p in ROOM_TYPES for p in parts):
        return 0.95
    if len(parts) > 1 and any(p in ROOM_TYPES for p in parts):
        return 0.7
    # Contains a known label as a substring, e.g. an OCR slip that kept most characters.
    for label in _KNOWN_LABELS:
        if len(label) >= 2 and label in normalised:
            return 0.5 * (len(label) / len(normalised))
    return 0.0
