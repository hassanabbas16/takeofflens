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
    "MAKUUHUONE": "bedroom",
    "MAKUUH": "bedroom",
    "OH": "living_room",       # olohuone, 99
    "OLOHUONE": "living_room",
    "OLOH": "living_room",
    "OLESKELU": "living_room",
    "H": "room",               # huone, 35
    "AH": "hobby_room",        # askarteluhuone, 10
    "ASKARTELU": "hobby_room",
    "ALKOVI": "alcove",
    # Kitchen / dining
    "K": "kitchen",            # keittiö, 73
    "KEITTIO": "kitchen",      # KEITTIÖ, 23
    "KK": "kitchen",           # keittokomero, 9
    "KT": "kitchen",           # keittotila
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
    "LO": "sauna",             # löylyhuone
    "LOYLYHUONE": "sauna",
    # Utility / storage
    "KHH": "utility",          # kodinhoitohuone, 33
    "VAR": "storage",          # varasto, 42
    "VARASTO": "storage",      # 12
    "TEKN": "technical_room",  # 8
    "TEKNINEN": "technical_room",
    "ULLAKKO": "attic",
    "APUK": "utility",         # apukeittiö
    "SK": "closet",            # siivouskomero
    "PK": "closet",
    # Closets / dressing
    "VH": "walk_in_closet",    # vaatehuone, 63
    "VAATEHUONE": "walk_in_closet",
    "PUKUH": "dressing_room",  # pukuhuone, 5
    "KOMERO": "closet",
    # Work
    "TYOHUONE": "office",      # TYÖHUONE
    "TYOTILA": "office",
    "TH": "office",            # 7
    "RT": "office",
    "LH": "room",              # lastenhuone
    # Outdoor
    "ULKOTILA": "outdoor",     # 131
    "PARVEKE": "balcony",      # 31
    "PARV": "balcony",
    "TERASSI": "terrace",      # 25
    "KUISTI": "porch",         # 17
    "AUTOKATOS": "carport",    # 9
    "AUTOVAJA": "carport",
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


_TRAILING_INDEX_RE = re.compile(r"^([A-Z]+)\d{1,2}$")


def lookup(text: str) -> str:
    """Return the English room type for a Finnish label, or ``unknown``.

    Falls back to stripping a trailing index, because plans number repeated rooms:
    ``MH1`` and ``MH2`` are both bedrooms. Only a trailing 1-2 digit suffix is stripped,
    so this cannot turn a dimension token into a room label.
    """
    normalised = normalise_label(text)
    if normalised in ROOM_TYPES:
        return ROOM_TYPES[normalised]
    indexed = _TRAILING_INDEX_RE.match(normalised)
    if indexed and indexed.group(1) in ROOM_TYPES:
        return ROOM_TYPES[indexed.group(1)]
    return UNKNOWN


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
    resolved = resolve_label(text)
    if resolved is not None and "+" not in normalise_label(text):
        return ROOM_TYPES[resolved]
    types = lookup_compound(text)
    for candidate in types:
        if candidate != UNKNOWN:
            return candidate
    return types[0] if types else UNKNOWN


# Minimum label length before a one-character slip is forgiven. Short labels are too dense
# in the lexicon for this to be safe: K, H, S, WC are all real and one edit apart.
_FUZZY_MIN_LEN = 5


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by at most one substitution, insertion or deletion."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(1 for x, y in zip(a, b, strict=True) if x != y) <= 1
    # One is longer: check it becomes the other by deleting a single character.
    longer, shorter = (a, b) if la > lb else (b, a)
    for i in range(len(longer)):
        if longer[:i] + longer[i + 1:] == shorter:
            return True
    return False


def resolve_label(text: str) -> str | None:
    """Resolve a possibly-misread label to a lexicon entry, or None.

    Exact match first, then a trailing index (MH1), then a single-character slip for longer
    labels. The last rule exists because OCR reliably confuses similar glyphs on these
    scans - KEITTIO came back as "KEITTLO" (I read as L) on a real plan, which the exact
    lookup rejected and which cost a real area. Restricted to labels of at least
    5 characters so that short, dense abbreviations are never guessed at.
    """
    normalised = normalise_label(text)
    if not normalised:
        return None
    if normalised in ROOM_TYPES:
        return normalised
    indexed = _TRAILING_INDEX_RE.match(normalised)
    if indexed and indexed.group(1) in ROOM_TYPES:
        return indexed.group(1)
    if len(normalised) >= _FUZZY_MIN_LEN:
        for candidate in _KNOWN_LABELS:
            if (
                len(candidate) >= _FUZZY_MIN_LEN
                and _edit_distance_at_most_one(normalised, candidate)
            ):
                return candidate
    return None


def is_known_label(text: str) -> bool:
    return resolve_label(text) is not None


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
    indexed = _TRAILING_INDEX_RE.match(normalised)
    if indexed and indexed.group(1) in ROOM_TYPES:
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


def label_key(text: str) -> str:
    """Normalised comparison key for a room label, tolerant of what models actually return.

    A model handed a token reading "khh 6.8" often echoes it whole into ``label_raw``.
    Comparing that naively gives "KHH68", which matches no reference label - that single
    detail made one approach score 1/12 on a plan where it had actually found the rooms.
    This strips a trailing area, resolves OCR near-misses, and falls back to the plain
    normalised form.
    """
    from app.pipeline.parse_dims import parse

    parsed = parse(text)
    candidate = parsed.label if parsed.label else text
    resolved = resolve_label(candidate)
    return resolved if resolved is not None else normalise_label(candidate)
