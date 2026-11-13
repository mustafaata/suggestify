"""Ultra-lightweight Standalone Suggestify Web App & API for Render.com.
Zero-DB, Pure Numpy In-Memory Engine (< 400 MB RSS footprint, float32 core precision).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import gc
import gzip
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import re
from socketserver import ThreadingMixIn
import sys
import threading
import time
import unicodedata
from urllib.parse import parse_qs, urlparse
import zlib

import numpy as np
import requests

PORT = int(os.environ.get("PORT", 10001))

THEME_LABELS = {
    "heartbreak_regret": "ayrılık, pişmanlık ve özlem",
    "love_romance": "romantik aşk, sevgi ve tutku",
    "joy_carefree": "mutluluk, neşe ve pozitif enerji",
    "melancholy_lonely": "yalnızlık, karanlık ve içsel hüzün",
    "anger_rebellion": "öfke, isyan ve sert tavır",
    "street_flex": "sokak kültürü, güç ve yüksek ego (flex)",
    "neutral": "Dengeli / nötr anlatım",
}
THEME_KEYS = [
    "heartbreak_regret",
    "love_romance",
    "joy_carefree",
    "melancholy_lonely",
    "anger_rebellion",
    "street_flex",
]

# Feature index order matching analyzer/songdna/features.py
FEATURE_NAMES = [
    "valence", "arousal", "happy", "sad", "party", "relaxed",
    "aggressive", "danceability", "acoustic", "electronic",
    "instrumental", "bpm", "mode",
]
F_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
BPM_I = F_IDX["bpm"]
FEATURE_PRIORS = np.array(
    [1.5, 1.3, 1.6, 1.6, 1.3, 1.2, 1.4, 1.3, 1.0, 0.7, 0.9, 0.45, 0.25],
    dtype=np.float32,
)
FEATURE_PHRASES = [
    ("valence", "neşeli, pozitif", "hüzünlü, buruk/karanlık"),
    ("arousal", "yüksek enerji", "sakin, düşük enerji"),
    ("happy", "mutluluk veren", "neşeden uzak, ciddi/duygusal"),
    ("sad", "hüzünlü, duygusal", "hüzünden uzak, ferah"),
    ("party", "parti ve eğlence havası", "sakin, içe dönük atmosfer"),
    ("relaxed", "dingin, slow/rahat akış", "hareketli, dinamik"),
    ("aggressive", "sert, agresif", "yumuşak tını"),
    ("danceability", "dans ettiren groove", "dansa pek uygun değil"),
    ("acoustic", "akustik doku", "modern prodüksiyon"),
    ("electronic", "elektronik altyapı", "organik enstrümanlar"),
    ("instrumental", "enstrümantal", "vokal odaklı"),
    ("bpm", "hızlı tempo", "yavaş/orta tempo"),
    ("mode", "majör tonalite", "minör tonalite"),
]

SIG_SPECS = [
    ("strings_symphonic", "güçlü yaylılar/orkestral dokunuş", 1.65),
    ("powerful_vocal", "güçlü vokal ve nakarat patlaması", 1.55),
    ("distorted_guitar", "sert elektro gitar riffleri ve rock davul", 0.95),
    ("rap_turntable_groove", "ritmik vokal/rap-rock akışı ve groove", 1.45),
    ("melodic_trap_808", "derin 808 bas ve melodik trap altyapısı", 1.40),
    ("piano_keys", "piyano ve tuşlu çalgılar dokusu", 1.45),
    ("acoustic_organic", "akustik/temiz gitar ve organik telliler", 1.35),
    ("synth_electronic", "synthesizer ve elektronik pad dokusu", 1.20),
    ("brass_funk_soul", "üflemeliler (brass/saksafon) ve funk groove", 1.35),
    ("oriental_folk_wind", "Doğu/Anadolu ezgisi ve nefesli/telli dokusu", 1.35),
    ("cinematic_reverb", "geniş reverb ve sinematik/dramatik atmosfer", 1.30),
]
SIG_PRIORS = np.array([s[2] for s in SIG_SPECS], dtype=np.float32)
SCENE_SIG_BLEND = 0.55
_SIG_FULL_LABELS = (
    "Güçlü yaylılar & orkestral/senfonik dokunuş",
    "Güçlü, patlayıcı vokal & nakarat enerjisi",
    "Sert elektro gitar riffleri & güçlü rock davul",
    "Nu-metal / rap-rock akışı, scratch & ritmik groove",
    "Derin 808 sub-bass & atmosferik melodik rap/trap altyapısı",
    "Piyano & tuşlu çalgılar odaklı altyapı",
    "Akustik / temiz gitar & sıcak organik telli dokusu",
    "Synthesizer, arpej & elektronik pad dokusu",
    "Üflemeliler (brass/saksafon/trompet) & funk/soul groove",
    "Doğu/Anadolu ezgisi & nefesli/içli enstrüman dokusu",
    "Geniş reverb & sinematik/dramatik atmosfer",
)
_SEMITONE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

_NOISE_JUNK_BIT = 1
_NOISE_LIVE_BIT = 2
_NOISE_REMIX_BIT = 4
_FLAG_TR_LYRICS_BIT = 8
_JUNK_TITLE_RX = re.compile(
    r"(?i)(?:\bkaraoke\b|made famous by|originally performed by|\bsoundalike\b|"
    r"\bbacking\s+track\b|\b8d\s*audio\b|\bsped\s*up\b|\bnightcore\b|"
    r"\bslowed\s*(?:\+|&|ve|and)?\s*reverb\b|\bringtone\b|[\(\[\-]\s*instrumental\s+version)"
)
_JUNK_ARTIST_RX = re.compile(
    r"(?i)(?:\bkaraoke\b|\bthe\s+tribute\b|\btribute\s+band\b|\btribute\s+co\b|"
    r"\btribute\s+all[- ]stars\b|\bsoundalike\b|\b8d\s+tunes\b|\bbacking\s+tracks?\b)"
)
_LIVE_RX = re.compile(
    r"(?i)(?:\((?:[^)]*\b(?:live|ao\s+vivo|en\s+vivo|canl[ıi]|unplugged|acoustic\s+session|concert|konser|session)\b[^)]*)\)"
    r"|\[(?:[^\]]*\b(?:live|ao\s+vivo|en\s+vivo|canl[ıi]|unplugged|concert|konser|session)\b[^\]]*)\]"
    r"|-\s*(?:live|canl[ıi]|unplugged|konser|ao\s+vivo|en\s+vivo)\b"
    r"|\blive\s+(?:at|in|from|on|@)\b|canl[ıi]\s+(?:performans|konser|kay[ıi]t)\b)"
)
_REMIX_RX = re.compile(
    r"(?i)(?:\((?:[^)]*\b(?:remix|rmx|bootleg|mashup|rework|refix|club\s*edit|(?:vip|club|extended|dub|dance|radio|wave|lp|vocal|acoustic|instrumental|house|techno|trance|lofi|chill(?:out)?|ambient|deep)\s*mix)\b[^)]*)\)"
    r"|\[(?:[^\]]*\b(?:remix|rmx|bootleg|mashup|rework|refix|(?:vip|club|extended|dub|radio|wave|lp|vocal|dance|chill(?:out)?|ambient|deep)\s*mix)\b[^\]]*)\]"
    r"|-\s*(?:remix|rmx|bootleg|mashup|(?:club|extended|radio|dance|chill(?:out)?)\s*mix)\b|\bremix\s*$)"
)
_INST_COVER_RX = re.compile(
    r"(?i)(?:[\(\[\-]\s*(?:enstr[uü]mantal|instrumental|karaoke|playback|beat\s*version|acapella)\b|\b(?:enstr[uü]mantal|karaoke)\b)"
)
_DZ_COVER_RX = re.compile(r"^https://cdn-images\.dzcdn\.net/images/cover/([0-9a-fA-F]{32})/250x250-000000-80-0-0\.jpg$")

# 19 sorted style-family columns produced by analyzer/songdna/recommend.py:_genre_families
P_FAMILY = {
    "blues": 0,
    "brass": 1,
    "childrens": 2,
    "classical": 3,
    "electronic_chill": 4,
    "electronic_club": 5,
    "folk_world": 6,
    "funk_disco": 7,
    "hip_hop": 8,
    "jazz": 9,
    "latin": 10,
    "non_music": 11,
    "pop_ballad": 12,
    "pop_indie": 13,
    "pop_mainstream": 14,
    "reggae": 15,
    "rnb_soul": 16,
    "rock": 17,
    "stage_screen": 18,
}

_PITCH_MAJOR_CAMELOT = {
    "Ab": 4, "G#": 4, "Eb": 5, "D#": 5, "Bb": 6, "A#": 6, "F": 7,
    "C": 8, "G": 9, "D": 10, "A": 11, "E": 12, "B": 1, "Cb": 1,
    "F#": 2, "Gb": 2, "C#": 3, "Db": 3,
}
_PITCH_MINOR_CAMELOT = {
    "F": 4, "C": 5, "G": 6, "D": 7, "A": 8, "E": 9, "B": 10,
    "F#": 11, "Gb": 11, "C#": 12, "Db": 12, "Ab": 1, "G#": 1,
    "Eb": 2, "D#": 2, "Bb": 3, "A#": 3,
}
_PITCH_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}


_P_FAMILY_LABELS = {
    0: "Caz / Blues",
    1: "Bando / Üflemeli",
    2: "Çocuk",
    3: "Klasik / Senfonik",
    4: "Elektronik / Chill",
    5: "Elektronik / Pop",
    6: "Halk / Etnik",
    7: "Funk / Disco",
    8: "Rap / Hip-Hop",
    9: "Caz / Blues",
    10: "Latin",
    11: "Akustik / Spoken",
    12: "Pop / Balad",
    13: "Akustik / Indie",
    14: "Pop",
    15: "Reggae / Dub",
    16: "R&B / Soul",
    17: "Rock / Alternatif",
    18: "Klasik / Pop",
}

_MODE_PRIORS = {
    # Order: valence, arousal, happy, sad, party, relaxed, aggressive, danceability, acoustic, electronic, instrumental, bpm, mode
    "emotional": np.array([2.35, 0.90, 2.40, 2.55, 0.45, 1.75, 1.55, 0.35, 1.45, 0.45, 0.85, 0.12, 0.95], dtype=np.float32),
    "balanced": np.array([1.05, 1.80, 1.15, 1.15, 1.45, 0.95, 1.25, 2.15, 0.90, 1.00, 0.80, 1.65, 0.25], dtype=np.float32),
    "local_scene": np.array([1.45, 1.35, 1.50, 1.55, 1.25, 1.15, 1.30, 1.35, 1.10, 0.65, 0.85, 0.55, 0.35], dtype=np.float32),
    "adventurous": np.array([1.60, 1.65, 1.50, 1.55, 1.20, 1.25, 1.40, 1.45, 1.00, 0.55, 0.90, 0.70, 0.40], dtype=np.float32),
}


def _mode_feature_priors(mode_name: str) -> np.ndarray:
    """Returns per-mode feature salience priors aligned with each discovery mode."""
    return _MODE_PRIORS.get(mode_name, FEATURE_PRIORS)


@dataclass
class Params:
    alpha: float = 0.18  # Timbre (EffNet + CLAP)
    zeta: float = 0.21   # Arrangement signature
    beta: float = 0.25   # Salience Audio DNA
    gamma: float = 0.14  # Genre & style family
    delta: float = 0.13  # Era identity
    eta: float = 0.03    # Lyrics theme & emotion
    omega: float = 0.06  # Camelot harmonic key flow
    p: float = 1.6
    w_floor: float = 0.10
    lam: float = 0.84
    pool: int = 70
    k: int = 8
    strict_region: bool = False
    force_region: str | None = None
    mode_name: str = "balanced"


MODES = {
    "emotional": Params(mode_name="emotional", alpha=0.21, zeta=0.14, beta=0.31, gamma=0.03, delta=0.01, eta=0.22, omega=0.12, lam=0.78, pool=80, strict_region=False),
    "balanced": Params(mode_name="balanced", alpha=0.18, zeta=0.20, beta=0.24, gamma=0.135, delta=0.125, eta=0.06, omega=0.06, lam=0.84, pool=70, strict_region=False),
    "local_scene": Params(mode_name="local_scene", alpha=0.22, zeta=0.21, beta=0.17, gamma=0.19, delta=0.10, eta=0.14, omega=0.05, lam=0.82, pool=75, strict_region=True, force_region="tr"),
    "safe": Params(mode_name="safe", alpha=0.18, zeta=0.22, beta=0.22, gamma=0.15, delta=0.12, eta=0.06, omega=0.05, lam=0.88, pool=60, strict_region=True),
    "adventurous": Params(mode_name="adventurous", alpha=0.28, zeta=0.18, beta=0.36, gamma=0.0, delta=0.0, eta=0.08, omega=0.08, lam=0.60, pool=105, strict_region=False),
}


def _tr_fold(text: str) -> str:
    """Folds Turkish diacritics/dotted-I and accents so case-insensitive search works consistently."""
    if not text:
        return ""
    s = text.replace("İ", "i").replace("I", "i").replace("ı", "i")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().strip()


def dedupe_key(artist: str, title: str) -> str:
    return f"{_tr_fold(artist)}::{_tr_fold(title)}"


print("⏳ [Render] Kompakt katalog yükleniyor (catalog_compact.npz)...")
t0 = time.time()
base_dir = os.path.dirname(__file__)
npz_path = os.path.join(base_dir, "catalog_compact.npz")
meta_path = os.path.join(base_dir, "catalog_meta.json.gz")

_FLOAT32_KEYS = {
    "F", "Z", "mu", "sigma", "S_z", "sig_mu", "sig_sigma",
    "L_themes", "P", "years", "vocal_gender",
}
_SQ8_KEYS = ("E", "C", "G", "L_vec")
_ROW_SCALES: dict[str, np.ndarray] = {}


def _quantize_sq8(arr: np.ndarray, chunk_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    n, d = arr.shape
    q = np.empty((n, d), dtype=np.int8)
    sc = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        blk = arr[s:e].astype(np.float32)
        scale = np.maximum(np.abs(blk).max(axis=1) / 127.0, 1e-8)
        sc[s:e] = scale
        q[s:e] = np.rint(blk / scale[:, None]).astype(np.int8)
    return q, sc


data = np.load(npz_path)
matrices = {}
_file_set = set(data.files)
for key in data.files:
    if key.endswith("_q") or key.endswith("_scale"):
        continue
    if key in _SQ8_KEYS:
        if f"{key}_q" in _file_set and f"{key}_scale" in _file_set:
            continue
        arr = data[key]
        q_arr, sc_arr = _quantize_sq8(arr)
        del arr
        matrices[key] = q_arr
        _ROW_SCALES[key] = sc_arr
    else:
        arr = data[key]
        matrices[key] = arr.astype(np.float32) if key in _FLOAT32_KEYS else arr
        del arr

for key in _SQ8_KEYS:
    qk, sk = f"{key}_q", f"{key}_scale"
    if qk in _file_set and sk in _file_set:
        matrices[key] = data[qk]
        _ROW_SCALES[key] = data[sk].astype(np.float32)

data.close()
del data, _file_set
gc.collect()

# Reconstruct raw signature matrix S cleanly in float32 for scene-relative scaling
matrices["S"] = matrices["S_z"] * matrices["sig_sigma"] + matrices["sig_mu"]

_N_TRACKS = len(matrices["ids"])
if "artist_ids" not in matrices:
    matrices["artist_ids"] = np.zeros(_N_TRACKS, dtype=np.int64)
if "rank_pct" not in matrices:
    matrices["rank_pct"] = np.zeros(_N_TRACKS, dtype=np.uint8)
if "noise_flags" not in matrices:
    matrices["noise_flags"] = np.zeros(_N_TRACKS, dtype=np.uint8)

_RANK_PCT = matrices["rank_pct"]
_NOISE_FLAGS = matrices["noise_flags"]

# Build CSR forward + reverse ListenerGraph (Personalized PageRank over artist_relations)
_GRAPH_NODES = matrices.get("graph_nodes", np.zeros(0, dtype=np.int64))
_GRAPH_FWD_PTR = matrices.get("graph_fwd_ptr", np.zeros(1, dtype=np.int32))
_GRAPH_FWD_IDX = matrices.get("graph_fwd_idx", np.zeros(0, dtype=np.int32))
_GRAPH_V = len(_GRAPH_NODES)
_GRAPH_NODE_MAP: dict[int, int] = {int(aid): i for i, aid in enumerate(_GRAPH_NODES)}
if _GRAPH_V > 0 and len(_GRAPH_FWD_IDX) > 0:
    _rev_deg = np.bincount(_GRAPH_FWD_IDX, minlength=_GRAPH_V).astype(np.int32)
    _GRAPH_REV_PTR = np.zeros(_GRAPH_V + 1, dtype=np.int32)
    np.cumsum(_rev_deg, out=_GRAPH_REV_PTR[1:])
    _GRAPH_REV_IDX = np.empty(len(_GRAPH_FWD_IDX), dtype=np.int32)
    _cursor = _GRAPH_REV_PTR[:-1].copy()
    for _u in range(_GRAPH_V):
        for _v in _GRAPH_FWD_IDX[_GRAPH_FWD_PTR[_u]: _GRAPH_FWD_PTR[_u + 1]]:
            _p = _cursor[_v]
            _GRAPH_REV_IDX[_p] = _u
            _cursor[_v] = _p + 1
    del _rev_deg, _cursor
else:
    _GRAPH_REV_PTR = np.zeros(_GRAPH_V + 1, dtype=np.int32)
    _GRAPH_REV_IDX = np.zeros(0, dtype=np.int32)

_TRACK_GRAPH_IDX = np.array(
    [_GRAPH_NODE_MAP.get(int(aid), -1) for aid in matrices["artist_ids"]],
    dtype=np.int32,
)

_TR_SEED_ARTISTS = frozenset({
    "tarkan", "kenan dogulu", "mustafa sandal", "serdar ortac", "gulsen",
    "sertab erener", "hande yener", "demet akalin", "bengu", "hadise",
    "ajda pekkan", "sezen aksu", "nazan oncel", "candan ercetin", "goksel",
    "nil karaibrahimgil", "yalin", "emre aydin", "gokhan turkmen", "murat boz",
    "murat dalkilic", "buray", "ilyas yalcintas", "oguzhan koc", "mabel matiz",
    "edis", "simge", "zeynep bastik", "aleyna tilki", "irem derici",
    "merve ozbey", "melek mosso", "melike sahin", "karsu", "gaye su akyol",
    "altin gun", "semicenk", "mert demir", "emir can igrek", "athena",
    "mor ve otesi", "duman", "manga", "teoman", "sebnem ferah", "ozlem tekin",
    "hayko cepkin", "pentagram", "kurban", "cilekes", "vega", "kargo", "mfo",
    "yuksek sadakat", "gripin", "model", "seksendort", "gece yolculari",
    "zakkum", "pinhani", "redd", "peyk", "adamlar", "buyuk ev ablukada",
    "yuzyuzeyken konusuruz", "dolu kadehi ters tut", "madrigal", "kofn",
    "perdenin ardindakiler", "yasli amca", "son feci bisiklet",
    "yok oyle kararli seyler", "kalben", "canozan", "sena sener", "nilipek.",
    "jakuzi", "she past away", "lin pesto", "ceza", "sagopa kajmer",
    "saniser", "gazapizm", "ezhel", "ben fero", "murda", "motive", "uzi",
    "lvbel c5", "blok3", "ati242", "cakal", "reckol", "sefo", "dedubluman",
    "canbay & wolker", "mavi", "poizi", "era7capone", "reynmen", "gunes",
    "baneva", "lil zey", "taladro", "sancak", "no.1", "ozbi", "mela bedel",
    "ikilem", "kurtulus kus", "burak bulut", "batuflex", "heijan", "wegh",
    "norm ender", "patron", "sehinsah", "defkhan", "summer cem",
})
_UNIQUELY_TR_CHARS = frozenset("ğğıİıışŞ")
_TR_OTHER_CHARS = frozenset("çÇöÖüÜ")
_TR_STOPWORDS = frozenset({
    "bir", "gibi", "degil", "yok", "ask", "gece", "yalan", "hayat", "kalp",
    "dunya", "ruya", "seni", "beni", "bana", "sana", "benim", "senin", "olsun",
    "olmaz", "neden", "simdi", "istanbul", "ankara", "kader", "aksam", "sabah",
    "goz", "soz", "ruzgar", "yagmur", "mavi", "sehir", "sokak", "hic", "hep",
    "cunku", "kadar", "bazen", "bugun", "yarin", "kim", "nasil", "nerede",
    "sevda", "hasret", "huzun", "mutlu", "agla", "gul", "sarki", "turku",
})
_NON_TR_WORDS = frozenset({
    "der", "die", "das", "und", "ich", "nicht", "kein", "mein", "dein", "sehr",
    "les", "des", "une", "pour", "avec", "dans", "mon", "ton", "vous", "nous",
    "est", "sont", "sur", "sous", "sans",
})


def _has_tr_text(text: str) -> bool:
    if any(c in _UNIQUELY_TR_CHARS or c in _TR_OTHER_CHARS for c in (text or "")):
        return True
    words = set(re.findall(r"[a-z]+", _tr_fold(text)))
    return len(words & _TR_STOPWORDS) >= 2 and not (words & _NON_TR_WORDS)


def _track_has_tr_evidence(title: str, artist: str) -> bool:
    text = f"{title} {artist}"
    if any(c in _UNIQUELY_TR_CHARS for c in text):
        return True
    words = set(re.findall(r"[a-z]+", _tr_fold(text)))
    if words & _NON_TR_WORDS:
        return False
    return bool((words & _TR_STOPWORDS) or any(c in _TR_OTHER_CHARS for c in text))


def _decode_cover(idx: int) -> str | None:
    if _COVER_HEX_OK[idx]:
        off = idx * 16
        return f"https://cdn-images.dzcdn.net/images/cover/{_COVER_HEX_BUF[off:off + 16].hex()}/250x250-000000-80-0-0.jpg"
    return _COVER_OTHER.get(idx)


def _top_sig_traits_for(idx: int, threshold: float = 0.45, max_items: int = 2) -> list[str]:
    sz = matrices["S_z"][idx]
    order = np.argsort(-sz)
    out = []
    for j in order:
        if float(sz[j]) >= threshold:
            out.append(_SIG_FULL_LABELS[int(j)])
            if len(out) >= max_items:
                break
    return out


_META_KEYS = (
    "id", "title", "artist_name", "year", "region", "cover_url",
    "preview", "bpm", "key", "scale", "lyric_dominant", "sig_traits",
)


class _TrackMeta(Mapping):
    """Zero-dict-overhead columnar Mapping view mimicking per-track metadata dicts."""
    __slots__ = ("_i",)

    def __init__(self, idx: int):
        self._i = idx

    def __len__(self) -> int:
        return len(_META_KEYS)

    def __iter__(self):
        return iter(_META_KEYS)

    def __getitem__(self, key: str):
        i = self._i
        if key == "id":
            return int(matrices["ids"][i])
        if key == "title":
            return _TITLES[i]
        if key == "artist_name":
            return _ARTISTS[i]
        if key == "year":
            y = int(_YEARS_RAW[i])
            return y if y > 0 else None
        if key == "region":
            return "tr" if _REG_CODE[i] == 1 else "global"
        if key == "cover_url":
            return _decode_cover(i)
        if key == "preview":
            return None
        if key == "bpm":
            b = float(_BPM_F16[i])
            return round(b, 1) if b > 0 else round(float(2.0 ** matrices["F"][i, BPM_I]), 1)
        if key == "key":
            p = int(CAMELOT_PITCH[i])
            return _SEMITONE_NAMES[p] if 0 <= p < 12 else None
        if key == "scale":
            md = int(CAMELOT_MODE[i])
            return "major" if md == 1 else ("minor" if md == 0 else None)
        if key == "lyric_dominant":
            if not matrices["has_lyrics"][i]:
                return None
            row = matrices["L_themes"][i]
            top_t = int(np.argmax(row))
            return THEME_KEYS[top_t] if float(row[top_t]) >= 0.32 else "neutral"
        if key == "sig_traits":
            return _top_sig_traits_for(i)
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            v = self[key]
            return default if v is None else v
        except KeyError:
            return default

    def __setitem__(self, key: str, val) -> None:
        if key == "region":
            _REG_CODE[self._i] = 1 if val == "tr" else 0
            return
        raise TypeError(f"_TrackMeta only supports updating 'region', got {key!r}")


def _iter_json_array(path: str):
    """Streams JSON array items one object at a time without allocating 60K transient dicts at once."""
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0
    started = False
    with gzip.open(path, "rt", encoding="utf-8") as f:
        while True:
            chunk = f.read(131072)
            eof = not chunk
            if chunk:
                buf = (buf[pos:] + chunk) if pos else (buf + chunk)
                pos = 0
            while True:
                n = len(buf)
                while pos < n and buf[pos] in " \r\n\t,":
                    pos += 1
                if not started:
                    if pos < n and buf[pos] == "[":
                        started = True
                        pos += 1
                        continue
                    break
                if pos >= n or buf[pos] == "]":
                    break
                try:
                    obj, next_pos = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield obj
                pos = next_pos
            if eof:
                break


_TITLES: list[str] = [""] * _N_TRACKS
_ARTISTS: list[str] = [""] * _N_TRACKS
_ART_FOLD: list[str] = [""] * _N_TRACKS
_TIT_FOLD: list[str] = [""] * _N_TRACKS
_COVER_HEX_BUF = bytearray(16 * _N_TRACKS)
_COVER_HEX_OK = np.zeros(_N_TRACKS, dtype=bool)
_COVER_OTHER: dict[int, str] = {}
_REG_CODE = np.zeros(_N_TRACKS, dtype=np.uint8)
_YEARS_RAW = np.zeros(_N_TRACKS, dtype=np.int16)
_BPM_F16 = np.zeros(_N_TRACKS, dtype=np.float16)
CAMELOT_NUM = np.zeros(_N_TRACKS, dtype=np.int8)
CAMELOT_MODE = np.full(_N_TRACKS, -1, dtype=np.int8)
CAMELOT_PITCH = np.full(_N_TRACKS, -1, dtype=np.int8)

_missing_nf = int(_NOISE_FLAGS.sum()) == 0
for idx, m in enumerate(_iter_json_array(meta_path)):
    title = m.get("title") or ""
    artist = sys.intern(m.get("artist_name") or "")
    _TITLES[idx] = title
    _ARTISTS[idx] = artist
    af = sys.intern(_tr_fold(artist))
    tf = _tr_fold(title)
    _ART_FOLD[idx] = af
    _TIT_FOLD[idx] = tf
    _REG_CODE[idx] = 1 if m.get("region") == "tr" else 0
    yr = m.get("year")
    if yr:
        _YEARS_RAW[idx] = int(yr)
    bp = m.get("bpm")
    if bp:
        _BPM_F16[idx] = np.float16(float(bp))
    cov = m.get("cover_url") or ""
    if cov:
        cm = _DZ_COVER_RX.match(cov)
        if cm:
            off = idx * 16
            _COVER_HEX_BUF[off:off + 16] = bytes.fromhex(cm.group(1))
            _COVER_HEX_OK[idx] = True
        else:
            _COVER_OTHER[idx] = cov
    k_str = (m.get("key") or "").strip()
    sc_str = (m.get("scale") or "").strip().lower()
    if k_str in _PITCH_SEMITONE:
        CAMELOT_PITCH[idx] = _PITCH_SEMITONE[k_str]
    if sc_str == "major" and k_str in _PITCH_MAJOR_CAMELOT:
        CAMELOT_NUM[idx] = _PITCH_MAJOR_CAMELOT[k_str]
        CAMELOT_MODE[idx] = 1
    elif sc_str == "minor" and k_str in _PITCH_MINOR_CAMELOT:
        CAMELOT_NUM[idx] = _PITCH_MINOR_CAMELOT[k_str]
        CAMELOT_MODE[idx] = 0
    if _missing_nf:
        nf = 0
        if _JUNK_TITLE_RX.search(title) or _JUNK_ARTIST_RX.search(artist):
            nf |= _NOISE_JUNK_BIT
        if _LIVE_RX.search(title):
            nf |= _NOISE_LIVE_BIT
        if _REMIX_RX.search(title):
            nf |= _NOISE_REMIX_BIT
        _NOISE_FLAGS[idx] = nf

gc.collect()


class _MetaSequence(Sequence):
    """Lazy zero-overhead Sequence returning columnar _TrackMeta views on demand."""
    def __len__(self) -> int:
        return _N_TRACKS

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [_TrackMeta(i) for i in range(*idx.indices(_N_TRACKS))]
        if idx < 0:
            idx += _N_TRACKS
        if idx < 0 or idx >= _N_TRACKS:
            raise IndexError(idx)
        return _TrackMeta(idx)


meta_list = _MetaSequence()


def _sanitize_catalog_regions() -> None:
    """Fixes BFS region spillover while restoring TR artists verified via lyrics or co-listening graph."""
    by_art_tr: dict[str, list[int]] = {}
    graph_tr_nodes: set[int] = set()

    for i in range(_N_TRACKS):
        af = _ART_FOLD[i]
        tid = int(matrices["ids"][i])
        if af == "motive" and tid < 500_000_000:
            _REG_CODE[i] = 0
            continue
        if af in _TR_SEED_ARTISTS and _GRAPH_V > 0 and _TRACK_GRAPH_IDX[i] >= 0:
            graph_tr_nodes.add(int(_TRACK_GRAPH_IDX[i]))
        if _REG_CODE[i] == 1:
            by_art_tr.setdefault(af, []).append(i)

    # Pass 1: Text diacritics, TR seed list & verified TR lyrics evidence
    ambiguous_artists = []
    for af, idxs in by_art_tr.items():
        if af in _TR_SEED_ARTISTS:
            continue
        ev_idxs = [
            i for i in idxs
            if _track_has_tr_evidence(_TITLES[i], _ARTISTS[i])
            or ((int(_NOISE_FLAGS[i]) & _FLAG_TR_LYRICS_BIT) != 0)
        ]
        has_non_tr_lang = any(
            set(re.findall(r"[a-z]+", _TIT_FOLD[i])) & _NON_TR_WORDS for i in idxs
        )
        is_tr_artist = (len(ev_idxs) >= 2 or (len(idxs) <= 2 and len(ev_idxs) == 1)) and not (
            has_non_tr_lang and len(ev_idxs) < 2
        )
        if is_tr_artist:
            if _GRAPH_V > 0:
                for i in idxs:
                    u = int(_TRACK_GRAPH_IDX[i])
                    if u >= 0:
                        graph_tr_nodes.add(u)
        else:
            ambiguous_artists.append((af, idxs, set(ev_idxs)))

    # Pass 2: Rescue genuine Turkish artists (ASCII names/titles) whose Deezer co-listening
    # neighborhood is strongly anchored in verified Turkish artists (no chaining beyond Pass 1)
    for af, idxs, ev_set in ambiguous_artists:
        has_non_tr_lang = any(
            set(re.findall(r"[a-z]+", _TIT_FOLD[i])) & _NON_TR_WORDS for i in idxs
        )
        graph_rescue = False
        if not has_non_tr_lang and _GRAPH_V > 0 and graph_tr_nodes and af != "motive":
            u_nodes = {int(_TRACK_GRAPH_IDX[i]) for i in idxs if _TRACK_GRAPH_IDX[i] >= 0}
            for u in u_nodes:
                nbrs = _GRAPH_FWD_IDX[_GRAPH_FWD_PTR[u]: _GRAPH_FWD_PTR[u + 1]]
                if len(nbrs) >= 3:
                    tr_hits = sum(1 for nb in nbrs if int(nb) in graph_tr_nodes)
                    if tr_hits >= 2 and (tr_hits / len(nbrs)) >= 0.35:
                        graph_rescue = True
                        break
        if graph_rescue:
            continue
        for i in idxs:
            if i not in ev_set or not any(c in _UNIQUELY_TR_CHARS for c in _TITLES[i]):
                _REG_CODE[i] = 0


_sanitize_catalog_regions()
print(f"✅ Katalog hazır: {_N_TRACKS} parça ({time.time() - t0:.2f}s)")

# Compact O(1) lookup structures saving ~140 MB Python heap overhead
_ID_SORT_ORDER = np.argsort(matrices["ids"], kind="mergesort")
_ID_SORTED = matrices["ids"][_ID_SORT_ORDER]


class _IdIndex(Mapping):
    def __len__(self) -> int:
        return _N_TRACKS

    def __iter__(self):
        return (int(x) for x in matrices["ids"])

    def __contains__(self, tid) -> bool:
        try:
            val = int(tid)
        except Exception:
            return False
        pos = int(np.searchsorted(_ID_SORTED, val))
        return pos < _N_TRACKS and int(_ID_SORTED[pos]) == val

    def __getitem__(self, tid: int):
        val = int(tid)
        pos = int(np.searchsorted(_ID_SORTED, val))
        if pos < _N_TRACKS and int(_ID_SORTED[pos]) == val:
            idx = int(_ID_SORT_ORDER[pos])
            return idx, meta_list[idx]
        raise KeyError(tid)


_BY_ARTIST_IDX: dict[str, list[int]] = {}
_BY_ARTIST_COMP_IDX: dict[str, list[int]] = {}
_art_id_map: dict[str, int] = {}
_ART_GROUP_ID = np.empty(_N_TRACKS, dtype=np.int32)
_CAT_KEY_HASH = np.empty(_N_TRACKS, dtype=np.uint64)

for idx in range(_N_TRACKS):
    af = _ART_FOLD[idx]
    tf = _TIT_FOLD[idx]
    _BY_ARTIST_IDX.setdefault(af, []).append(idx)
    _BY_ARTIST_COMP_IDX.setdefault(af.replace(" ", ""), []).append(idx)
    aid_grp = _art_id_map.get(af)
    if aid_grp is None:
        aid_grp = len(_art_id_map)
        _art_id_map[af] = aid_grp
    _ART_GROUP_ID[idx] = aid_grp
    t_crc = zlib.crc32(tf.encode("utf-8"))
    _CAT_KEY_HASH[idx] = (np.uint64(aid_grp) << np.uint64(32)) | np.uint64(t_crc)
del _art_id_map


class _KeyIndex(Mapping):
    def __len__(self) -> int:
        return _N_TRACKS

    def __iter__(self):
        return (f"{_ART_FOLD[i]}::{_TIT_FOLD[i]}" for i in range(_N_TRACKS))

    def _find(self, key: str) -> int | None:
        if "::" not in key:
            return None
        af, tf = key.split("::", 1)
        for idx in _BY_ARTIST_IDX.get(af, ()):
            if _TIT_FOLD[idx] == tf:
                return idx
        return None

    def __contains__(self, key) -> bool:
        return isinstance(key, str) and self._find(key) is not None

    def __getitem__(self, key: str):
        idx = self._find(key)
        if idx is None:
            raise KeyError(key)
        return idx, meta_list[idx]


class _ArtistIndex(Mapping):
    def __init__(self, idx_map: dict[str, list[int]]):
        self._m = idx_map

    def __len__(self) -> int:
        return len(self._m)

    def __iter__(self):
        return iter(self._m)

    def __contains__(self, key) -> bool:
        return key in self._m

    def __getitem__(self, key: str):
        return [(i, meta_list[i]) for i in self._m[key]]


by_id = _IdIndex()
by_key = _KeyIndex()
by_artist = _ArtistIndex(_BY_ARTIST_IDX)
by_artist_compact = _ArtistIndex(_BY_ARTIST_COMP_IDX)

# Thread-safe TTL caches for media hydration, online search, and related artist lookup
_MEDIA_CACHE_LOCK = threading.Lock()
_MEDIA_CACHE: dict[int, tuple[str | None, str | None, float]] = {}
_MEDIA_CACHE_TTL = 900.0  # 15 minutes
_MEDIA_CACHE_MAX = 2500

_SEARCH_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE: dict[tuple[str, int, bool], tuple[list[dict], float]] = {}
_RELATED_CACHE: dict[int, tuple[list[dict], float]] = {}
_SEARCH_CACHE_TTL = 300.0
_SEARCH_CACHE_MAX = 500


def _cache_media_get(tid: int) -> tuple[str | None, str | None] | None:
    now = time.monotonic()
    with _MEDIA_CACHE_LOCK:
        entry = _MEDIA_CACHE.get(tid)
        if entry and entry[2] > now:
            return entry[0], entry[1]
        if entry:
            _MEDIA_CACHE.pop(tid, None)
    return None


def _cache_media_put(tid: int, cover_url: str | None, preview_url: str | None) -> None:
    now = time.monotonic()
    with _MEDIA_CACHE_LOCK:
        if len(_MEDIA_CACHE) >= _MEDIA_CACHE_MAX:
            expired = [k for k, v in _MEDIA_CACHE.items() if v[2] <= now]
            for k in expired:
                _MEDIA_CACHE.pop(k, None)
            if len(_MEDIA_CACHE) >= _MEDIA_CACHE_MAX:
                _MEDIA_CACHE.pop(next(iter(_MEDIA_CACHE)), None)
        _MEDIA_CACHE[tid] = (cover_url, preview_url, now + _MEDIA_CACHE_TTL)


def _infer_online_region(artist_name: str, title: str, from_tr_store: bool = False) -> str:
    af = _tr_fold(artist_name)
    if af in _TR_SEED_ARTISTS:
        return "tr"
    if af in by_artist and any(m.get("region") == "tr" for _, m in by_artist[af]):
        # Avoid falsely tagging Ariana Grande / US homonym via title-only match
        if af != "motive" or _has_tr_text(title):
            return "tr"
    if _has_tr_text(f"{artist_name} {title}"):
        return "tr"
    return "tr" if (from_tr_store and any(c in _UNIQUELY_TR_CHARS for c in f"{artist_name} {title}")) else "global"


def deezer_search(q: str):
    return search_online_tracks(q, limit=5)


def search_online_tracks(q: str, limit: int = 5, prefer_tr: bool = False):
    clean_q = q.replace("—", " - ").replace("–", " - ").replace("_", " ").strip()
    q_fold = _tr_fold(clean_q)
    if not q_fold:
        return []

    cache_key = (q_fold, int(limit), bool(prefer_tr))
    now = time.monotonic()
    with _SEARCH_CACHE_LOCK:
        cached = _SEARCH_CACHE.get(cache_key)
        if cached and cached[1] > now:
            return [dict(item) for item in cached[0]]

    results = []
    seen_ids = set()
    seen_keys = set()

    # 1. Deezer API search
    try:
        url = f"https://api.deezer.com/search?q={requests.utils.quote(clean_q)}&limit={limit}"
        r = requests.get(url, timeout=3.5)
        for t in r.json().get("data", []):
            tid = int(t["id"])
            title = t.get("title_short") or t.get("title") or ""
            art_obj = t.get("artist") or {}
            artist = art_obj.get("name", "Bilinmeyen")
            dkey = dedupe_key(artist, title)
            if tid in seen_ids or dkey in seen_keys:
                continue
            seen_ids.add(tid)
            seen_keys.add(dkey)
            cov = (t.get("album") or {}).get("cover_medium")
            prev = t.get("preview")
            _cache_media_put(tid, cov, prev)
            rel_date = (t.get("album") or {}).get("release_date") or "2023"
            results.append({
                "id": tid,
                "artist_id": art_obj.get("id"),
                "title": title,
                "artist_name": artist,
                "genre": "Alternatif / Pop",
                "preview": prev,
                "cover_url": cov,
                "year": int(rel_date[:4]) if str(rel_date)[:4].isdigit() else 2023,
                "region": _infer_online_region(artist, title),
            })
    except Exception:
        pass

    # 2. iTunes TR store fallback/enrichment (essential for Turkish homonyms like Motive & regional releases)
    need_itunes = len(results) < limit or prefer_tr or not any(item["region"] == "tr" for item in results)
    if need_itunes:
        try:
            url = f"https://itunes.apple.com/search?term={requests.utils.quote(clean_q)}&country=TR&entity=song&limit={limit}"
            r = requests.get(url, timeout=3.5)
            for item in r.json().get("results", []):
                tid = int(item["trackId"])
                title = item.get("trackName") or ""
                artist = item.get("artistName") or "Bilinmeyen"
                dkey = dedupe_key(artist, title)
                if tid in seen_ids or dkey in seen_keys:
                    continue
                seen_ids.add(tid)
                seen_keys.add(dkey)
                rel_date = item.get("releaseDate") or "2023"
                results.append({
                    "id": tid,
                    "title": title,
                    "artist_name": artist,
                    "genre": item.get("primaryGenreName") or "Pop",
                    "preview": item.get("previewUrl"),
                    "cover_url": (item.get("artworkUrl100") or "").replace("100x100bb", "300x300bb"),
                    "year": int(rel_date[:4]) if str(rel_date)[:4].isdigit() else 2023,
                    "region": _infer_online_region(artist, title, from_tr_store=True),
                })
        except Exception:
            pass

    out = results[: max(limit, 8)] if (prefer_tr and len(results) > limit) else results[:limit]
    if prefer_tr:
        results.sort(key=lambda it: (0 if it["region"] == "tr" else 1))
        out = results[:limit]

    with _SEARCH_CACHE_LOCK:
        if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX:
            _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)), None)
        _SEARCH_CACHE[cache_key] = ([dict(x) for x in out], now + _SEARCH_CACHE_TTL)

    return out


def _pick_artist_rep_track(
    tracks: list[tuple[int, dict]],
    prefer_tr: bool = False,
    target_year: int | None = None,
) -> tuple[int, dict]:
    """Selects a clean studio representative track for an artist, prioritizing popularity & era."""
    clean = [(i, m) for i, m in tracks if (int(_NOISE_FLAGS[i]) & 7) == 0]
    pool = clean if clean else tracks
    tr_tracks = [(i, m) for i, m in pool if _REG_CODE[i] == 1]
    if tr_tracks and (prefer_tr or len(tr_tracks) >= len(pool) * 0.4):
        pool = tr_tracks

    sz = matrices["S_z"]
    if target_year is not None:
        return min(
            pool,
            key=lambda pair: (
                int(_NOISE_FLAGS[pair[0]]) & 7,
                abs(int(pair[1].get("year") or target_year) - int(target_year)),
                -int(_RANK_PCT[pair[0]]),
                0 if max(float(sz[pair[0], 0]), float(sz[pair[0], 4]), float(sz[pair[0], 5]), float(sz[pair[0], 6])) >= 0.45 else 1,
                int(matrices["ids"][pair[0]]),
            ),
        )
    # Prefer well-known clean studio tracks with clear arrangement signature
    return max(
        pool,
        key=lambda pair: (
            1 if (int(_NOISE_FLAGS[pair[0]]) & 7) == 0 else 0,
            int(_RANK_PCT[pair[0]]),
            1 if max(float(sz[pair[0], 0]), float(sz[pair[0], 4]), float(sz[pair[0], 5]), float(sz[pair[0], 6])) >= 0.45 else 0,
            int(matrices["years"][pair[0]] or 2015),
        ),
    )


def _find_deezer_artist_proxy(artist_name: str, artist_id: int | None = None, prefer_tr: bool = False) -> tuple[int, dict] | None:
    """Resolves an out-of-catalog artist via Deezer related artists into the closest catalog proxy."""
    try:
        aid = int(artist_id) if artist_id else None
        if not aid:
            r = requests.get(
                f"https://api.deezer.com/search/artist?q={requests.utils.quote(artist_name)}&limit=3",
                timeout=3.0,
            )
            if r.status_code == 200:
                target_fold = _tr_fold(artist_name)
                candidates = r.json().get("data", [])
                for cand in candidates:
                    if _tr_fold(cand.get("name", "")) == target_fold:
                        aid = int(cand["id"])
                        break
                if not aid and candidates:
                    aid = int(candidates[0]["id"])
        if not aid:
            return None

        now = time.monotonic()
        related = None
        with _SEARCH_CACHE_LOCK:
            entry = _RELATED_CACHE.get(aid)
            if entry and entry[1] > now:
                related = entry[0]

        if related is None:
            rr = requests.get(f"https://api.deezer.com/artist/{aid}/related?limit=15", timeout=3.0)
            related = rr.json().get("data", []) if rr.status_code == 200 else []
            with _SEARCH_CACHE_LOCK:
                if len(_RELATED_CACHE) >= _SEARCH_CACHE_MAX:
                    _RELATED_CACHE.pop(next(iter(_RELATED_CACHE)), None)
                _RELATED_CACHE[aid] = (related, now + _SEARCH_CACHE_TTL)

        tr_match = None
        any_match = None
        for rel in related:
            rel_fold = _tr_fold(rel.get("name") or "")
            if rel_fold in by_artist:
                pick = _pick_artist_rep_track(by_artist[rel_fold], prefer_tr=prefer_tr)
                if pick[1].get("region") == "tr" and tr_match is None:
                    tr_match = pick
                if any_match is None:
                    any_match = pick
        return (tr_match or any_match) if prefer_tr else (any_match or tr_match)
    except Exception:
        return None


def deezer_track(tid: int):
    try:
        r = requests.get(f"https://api.deezer.com/track/{tid}", timeout=2.2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def _itunes_media_fallback(artist_name: str, title: str) -> tuple[str | None, str | None]:
    if not artist_name or not title:
        return None, None
    try:
        q = f"{artist_name} {title}".strip()
        r = requests.get(
            f"https://itunes.apple.com/search?term={requests.utils.quote(q)}&country=TR&entity=song&limit=3",
            timeout=2.2,
        )
        if r.status_code != 200:
            return None, None
        target_a = _tr_fold(artist_name)
        for item in r.json().get("results", []):
            prev = item.get("previewUrl")
            if not prev:
                continue
            item_a = _tr_fold(item.get("artistName") or "")
            if target_a in item_a or item_a in target_a:
                cov = (item.get("artworkUrl100") or "").replace("100x100bb", "300x300bb") or None
                return cov, prev
    except Exception:
        pass
    return None, None


def hydrate_track_media(
    tid: int,
    existing_cover: str | None = None,
    existing_preview: str | None = None,
    artist_name: str = "",
    title: str = "",
) -> tuple[str | None, str | None]:
    """Returns (cover_url, preview) using TTL cache, Deezer API, and iTunes TR preview fallback."""
    if existing_cover and existing_preview:
        return existing_cover, existing_preview
    cached = _cache_media_get(tid)
    if cached is not None:
        cov = existing_cover or cached[0]
        prev = existing_preview or cached[1]
        if cov or prev:
            return cov, prev
    dt = deezer_track(tid)
    cov = existing_cover or (dt.get("album") or {}).get("cover_medium") or None
    prev = existing_preview or dt.get("preview") or None
    if not prev and artist_name and title:
        it_cov, it_prev = _itunes_media_fallback(artist_name, title)
        cov = cov or it_cov
        prev = it_prev or prev
    _cache_media_put(tid, cov, prev)
    return cov, prev


def hydrate_media_batch(items: list[tuple]) -> dict[int, tuple[str | None, str | None]]:
    """Bounded parallel hydration of cover_url and preview across seed & recommendations."""
    results: dict[int, tuple[str | None, str | None]] = {}
    to_fetch: list[tuple[int, str | None, str | None, str, str]] = []
    for entry in items:
        tid, cov, prev = entry[0], entry[1], entry[2]
        art = entry[3] if len(entry) > 3 else ""
        tit = entry[4] if len(entry) > 4 else ""
        if cov and prev:
            results[tid] = (cov, prev)
            continue
        cached = _cache_media_get(tid)
        if cached is not None and (cached[0] or cached[1]):
            results[tid] = (cov or cached[0], prev or cached[1])
            continue
        to_fetch.append((tid, cov, prev, art, tit))

    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(9, len(to_fetch))) as pool:
            futs = [
                (tid, pool.submit(hydrate_track_media, tid, cov, prev, art, tit))
                for tid, cov, prev, art, tit in to_fetch
            ]
            for tid, fut in futs:
                try:
                    results[tid] = fut.result(timeout=4.0)
                except Exception:
                    results[tid] = (None, None)
    return results


def _mat_row(key: str, idx: int) -> np.ndarray:
    row = matrices[key][idx].astype(np.float32)
    sc = _ROW_SCALES.get(key)
    return row * sc[idx] if sc is not None else row


def _mat_rows(key: str, indices) -> np.ndarray:
    sub = matrices[key][indices].astype(np.float32)
    sc = _ROW_SCALES.get(key)
    return sub * sc[indices, None] if sc is not None else sub


def _chunk_dot(
    mat: np.ndarray,
    vec32: np.ndarray,
    chunk_size: int = 4096,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """Computes matrix-vector dot product in float32 chunks with optional SQ8 per-row scales."""
    n = len(mat)
    out = np.empty(n, dtype=np.float32)
    v = np.asarray(vec32, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        res = mat[start:end].astype(np.float32) @ v
        if scales is not None:
            res *= scales[start:end]
        out[start:end] = res
    return out


def _l2_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-9)


def _compute_sig_weights(sig_z: np.ndarray) -> np.ndarray:
    pos = np.clip(sig_z - 0.32, 0.0, 1.35)
    priors = SIG_PRIORS.copy()
    if sig_z[0] >= 0.65 or sig_z[5] >= 0.65 or sig_z[8] >= 0.65 or sig_z[9] >= 0.65:
        priors[2] *= 0.45
    raw_w = priors * (pos ** 1.45 + 0.02)
    w = raw_w / (raw_w.sum() + 1e-9)
    for _ in range(4):
        over = w > 0.45
        if not over.any():
            break
        excess = float((w[over] - 0.45).sum())
        w[over] = 0.45
        under = ~over & (pos > 0)
        if not under.any():
            break
        w[under] += excess * w[under] / (w[under].sum() + 1e-9)
    return (w / (w.sum() + 1e-9)).astype(np.float32)


def _distances_compact(seed_f: np.ndarray, seed_z: np.ndarray, F: np.ndarray, Z: np.ndarray) -> np.ndarray:
    raw_gap = np.abs(Z - seed_z[None, :]) / 2.5
    notable = np.abs(seed_z) > 0.40
    same_dir_stronger = (Z * seed_z[None, :] > 0) & (np.abs(Z) >= np.abs(seed_z)[None, :]) & notable[None, :]
    same_dir_solid = (Z * seed_z[None, :] > 0) & (np.abs(Z) >= 0.45) & notable[None, :]
    opposite = (Z * seed_z[None, :] < 0) & notable[None, :]
    D = np.where(
        same_dir_stronger,
        raw_gap * 0.28,
        np.where(same_dir_solid, raw_gap * 0.48, np.where(opposite, raw_gap * 1.60, raw_gap)),
    )
    diff = np.abs(F[:, BPM_I] - seed_f[BPM_I])
    octave = np.minimum(np.abs(diff - 1.0), np.abs(diff + 1.0)) + 0.12
    D[:, BPM_I] = np.clip(np.minimum(diff, octave) / 0.45, 0.0, 1.0)
    return np.clip(D, 0.0, 1.0).astype(np.float32)


def _mood_polarity_penalty_compact(seed_f: np.ndarray, F: np.ndarray) -> np.ndarray:
    pen = np.zeros(len(F), dtype=np.float32)
    s_val = float(seed_f[F_IDX["valence"]])
    s_happy = float(seed_f[F_IDX["happy"]])
    s_sad = float(seed_f[F_IDX["sad"]])
    s_aggr = float(seed_f[F_IDX["aggressive"]])
    s_dance = float(seed_f[F_IDX["danceability"]])

    c_sad = F[:, F_IDX["sad"]]
    c_happy = F[:, F_IDX["happy"]]
    c_party = F[:, F_IDX["party"]]
    c_aggr = F[:, F_IDX["aggressive"]]
    c_dance = F[:, F_IDX["danceability"]]

    if s_sad >= 0.42 and s_happy <= 0.25:
        pen += 1.5 * np.maximum(0.0, c_happy - 0.28) + 1.2 * np.maximum(0.0, c_party - 0.36)
    if s_sad >= 0.62 and s_dance <= 0.25:
        pen += 1.1 * np.maximum(0.0, c_dance - 0.48)
    if s_happy >= 0.55 and s_val >= 0.60:
        pen += 1.5 * np.maximum(0.0, c_sad - 0.36) + 1.3 * np.maximum(0.0, c_aggr - 0.32)
    if s_aggr <= 0.15:
        pen += 1.1 * np.maximum(0.0, c_aggr - 0.40)
    return pen


def _lyric_similarity_compact(seed_idx: int) -> tuple[np.ndarray, bool, float]:
    N = len(meta_list)
    out = np.zeros(N, dtype=np.float32)
    if not matrices["has_lyrics"][seed_idx]:
        return out, False, 0.0

    s_th = matrices["L_themes"][seed_idx]
    s_max = float(s_th.max())
    if s_max < 0.25:
        return out, False, s_max

    s_th_norm = s_th / (np.linalg.norm(s_th) + 1e-9)
    theme_cos = _l2_rows(matrices["L_themes"]) @ s_th_norm
    s_lv = _mat_row("L_vec", seed_idx)
    if np.linalg.norm(s_lv) > 1e-6:
        vec_cos = _chunk_dot(matrices["L_vec"], s_lv, scales=_ROW_SCALES.get("L_vec"))
        raw_sim = 0.68 * theme_cos + 0.32 * vec_cos
    else:
        raw_sim = theme_cos

    s_heartbreak = float(s_th[0])
    s_love = float(s_th[1])
    s_joy = float(s_th[2])
    s_mel = float(s_th[3])
    s_anger = float(s_th[4])
    s_flex = float(s_th[5])

    c_hb = matrices["L_themes"][:, 0]
    c_love = matrices["L_themes"][:, 1]
    c_joy = matrices["L_themes"][:, 2]
    c_mel = matrices["L_themes"][:, 3]
    c_anger = matrices["L_themes"][:, 4]
    c_flex = matrices["L_themes"][:, 5]

    clash = np.zeros(N, dtype=np.float32)
    if max(s_heartbreak, s_mel, s_anger) >= 0.45 and s_flex < 0.38 and s_joy < 0.30:
        clash += 0.72 * np.maximum(0.0, c_flex - max(s_flex, 0.22))
        clash += 0.60 * np.maximum(0.0, c_joy - 0.36)
    if s_heartbreak >= 0.55 and s_anger < 0.40:
        clash += 0.38 * np.maximum(0.0, c_anger - np.maximum(c_hb, 0.38))
    if s_heartbreak >= 0.55:
        clash += 0.90 * np.maximum(0.0, 0.50 - np.maximum(c_hb, c_mel))
    if max(s_heartbreak, s_anger) >= 0.55 and s_love < 0.45:
        clash += 0.42 * np.maximum(0.0, c_love - np.maximum(c_hb, c_anger) - 0.30)
    if s_joy >= 0.55 and s_heartbreak < 0.35:
        clash += 0.65 * np.maximum(0.0, np.maximum(c_hb, c_mel) - 0.40)
        clash += 0.55 * np.maximum(0.0, c_anger - 0.42)
    if s_love >= 0.50 and s_flex < 0.35:
        clash += 0.50 * np.maximum(0.0, c_flex - c_love)

    valid = matrices["has_lyrics"]
    out[valid] = np.clip(raw_sim[valid] * 1.08 - clash[valid], -0.8, 1.0)
    if valid.any():
        # Neutral baseline for tracks lacking LRCLIB lyrics (never punish missing coverage)
        out[~valid] = 0.34 if s_max >= 0.55 else 0.38
    return out, s_max >= 0.28, s_max


def _camelot_similarity(seed_idx: int) -> np.ndarray:
    """Computes harmonic Circle-of-Fifths (Camelot wheel) compatibility in [0.18, 1.0]."""
    n = len(meta_list)
    s_num = int(CAMELOT_NUM[seed_idx])
    s_mode = int(CAMELOT_MODE[seed_idx])
    s_pitch = int(CAMELOT_PITCH[seed_idx])
    if s_num <= 0 or s_mode < 0:
        return np.full(n, 0.50, dtype=np.float32)

    valid = (CAMELOT_NUM > 0) & (CAMELOT_MODE >= 0)
    diff = np.minimum((CAMELOT_NUM - s_num) % 12, (s_num - CAMELOT_NUM) % 12)
    same_mode = CAMELOT_MODE == s_mode

    sim = np.full(n, 0.18, dtype=np.float32)
    sim = np.where(valid & (diff == 2) & same_mode, 0.55, sim)
    if s_pitch >= 0:
        sim = np.where(valid & (CAMELOT_PITCH == s_pitch) & ~same_mode, 0.72, sim)
    sim = np.where(valid & (diff == 1) & same_mode, 0.85, sim)
    sim = np.where(valid & (diff == 0) & ~same_mode, 0.85, sim)
    sim = np.where(valid & (diff == 0) & same_mode, 1.00, sim)
    return sim


def _style_family_compatibility_compact(seed_idx: int, raw_genre_sim: np.ndarray) -> np.ndarray:
    """Ported V4 cross-family style guards (Hip Hop vs Rock/Acoustic, Trap vs Boom Bap, Club vs Folk)."""
    P = matrices["P"]
    S_z = matrices["S_z"]
    F = matrices["F"]
    years = matrices["years"]

    s_p = P[seed_idx]
    top_p = int(np.argmax(s_p))
    penalty = np.zeros(len(P), dtype=np.float32)

    if top_p == P_FAMILY["hip_hop"]:
        urban_mass = np.maximum(P[:, P_FAMILY["hip_hop"]], 0.78 * P[:, P_FAMILY["rnb_soul"]])
        penalty += 0.45 * np.maximum(0.0, 0.35 - urban_mass)
        for mis_key in ("rock", "folk_world", "pop_indie"):
            penalty += 0.30 * P[:, P_FAMILY[mis_key]]
        # Separate modern melodic trap / 808 era from strict old-school boom-bap rap-rock
        if float(S_z[seed_idx, 4]) >= 0.45 and float(years[seed_idx] or 2018) >= 2016 and float(S_z[seed_idx, 3]) < 0.30:
            penalty += 0.18 * np.clip((S_z[:, 3] - S_z[:, 4] - 0.60) / 1.4, 0.0, 1.0)
    elif top_p == P_FAMILY["rock"]:
        penalty += 0.45 * np.maximum(0.0, 0.35 - P[:, P_FAMILY["rock"]])
        penalty += 0.35 * P[:, P_FAMILY["hip_hop"]]
        if float(F[seed_idx, F_IDX["aggressive"]]) >= 0.55 and float(F[seed_idx, F_IDX["happy"]]) <= 0.25:
            penalty += 0.25 * P[:, P_FAMILY["folk_world"]]
            penalty += 0.15 * np.maximum(0.0, P[:, P_FAMILY["electronic_club"]] - 0.35)
    elif top_p in (P_FAMILY["electronic_club"], P_FAMILY["funk_disco"]) and (
        float(F[seed_idx, F_IDX["electronic"]]) >= 0.50 or float(F[seed_idx, F_IDX["danceability"]]) >= 0.65
    ):
        if float(s_p[P_FAMILY["rock"]]) < 0.25:
            penalty += 0.25 * np.maximum(0.0, P[:, P_FAMILY["rock"]] - 0.40)
        penalty += 0.25 * P[:, P_FAMILY["folk_world"]] * np.maximum(0.0, F[:, F_IDX["acoustic"]] - 0.55)

    if float(F[seed_idx, F_IDX["acoustic"]]) >= 0.68 and top_p != P_FAMILY["hip_hop"]:
        penalty += 0.40 * (P[:, P_FAMILY["hip_hop"]] + P[:, P_FAMILY["electronic_club"]])
    elif top_p in (P_FAMILY["pop_ballad"], P_FAMILY["pop_indie"], P_FAMILY["pop_mainstream"], P_FAMILY["jazz"]):
        if float(s_p[P_FAMILY["hip_hop"]]) < 0.15 and float(S_z[seed_idx, 3]) < 0.20:
            penalty += 0.30 * P[:, P_FAMILY["hip_hop"]]
        if float(F[seed_idx, F_IDX["electronic"]]) < 0.25 and float(F[seed_idx, F_IDX["danceability"]]) < 0.55:
            penalty += 0.32 * P[:, P_FAMILY["electronic_club"]]

    return (raw_genre_sim - penalty).astype(np.float32)


def _explain_compact(
    seed_idx: int,
    seed_sig_z: np.ndarray,
    seed_sig_w: np.ndarray,
    seed_w: np.ndarray,
    cz_sig: np.ndarray,
    cand_idx: int,
    drow: np.ndarray,
    era_val: float,
    tex_val: float,
    cam_val: float = 0.5,
    mode_name: str = "balanced",
    seed_top_p: int = -1,
    cand_top_p: int = -1,
    behav_val: float = 0.0,
) -> list[str]:
    out: list[str] = []

    if mode_name == "adventurous" and seed_top_p >= 0 and cand_top_p >= 0 and cand_top_p != seed_top_p:
        fam_name = _P_FAMILY_LABELS.get(cand_top_p)
        if fam_name:
            out.append(f"türler arası köprü: {fam_name}")

    sig_matches: list[tuple[float, str]] = []
    for j, (key, short, _prior) in enumerate(SIG_SPECS):
        if seed_sig_z[j] >= 0.50 and cz_sig[cand_idx, j] >= 0.25:
            hook_boost = 0.65 if key == "distorted_guitar" else 1.15
            strength = hook_boost * float(seed_sig_w[j]) * min(float(seed_sig_z[j]), float(cz_sig[cand_idx, j]) + 0.55)
            sig_matches.append((strength, short))
    sig_matches.sort(reverse=True)
    if sig_matches:
        out.append(f"imza: {' + '.join(s for _, s in sig_matches[:2])}")

    sz = matrices["Z"][seed_idx]
    cz = matrices["Z"][cand_idx]
    cf = matrices["F"][cand_idx]
    dna_cands = []
    for j, (fname, high_p, low_p) in enumerate(FEATURE_PHRASES):
        if abs(float(sz[j])) < 0.35 or np.sign(cz[j]) != np.sign(sz[j]):
            continue
        p = high_p if sz[j] > 0 else low_p
        if fname == "bpm":
            bpm_val = meta_list[cand_idx].get("bpm") or (2.0 ** float(cf[j]))
            p = f"{p} (~{float(bpm_val):.0f} BPM)"
        dna_cands.append((float(seed_w[j]) * (1.0 - float(drow[j])), p))
    dna_cands.sort(reverse=True)
    if mode_name == "balanced":
        bpm_val = meta_list[cand_idx].get("bpm") or (2.0 ** float(cf[BPM_I]))
        if float(drow[BPM_I]) <= 0.30 and not any("BPM" in p for _, p in dna_cands[:2]):
            out.append(f"tempo uyumu (~{float(bpm_val):.0f} BPM)")
    out.extend(p for _, p in dna_cands[:2])

    if matrices["has_lyrics"][seed_idx] and matrices["has_lyrics"][cand_idx]:
        s_th = matrices["L_themes"][seed_idx]
        c_th = matrices["L_themes"][cand_idx]
        shared = np.where(s_th >= 0.42, np.minimum(s_th, c_th), 0.0)
        best_t = int(np.argmax(shared))
        if shared[best_t] >= 0.36:
            t_key = THEME_KEYS[best_t]
            lyr_phrase = f"söz teması: {THEME_LABELS[t_key]}"
            if mode_name == "emotional":
                out.insert(0, lyr_phrase)
            else:
                out.append(lyr_phrase)

    if behav_val >= 0.42 and mode_name in ("balanced", "local_scene", "safe") and len(out) < 4:
        out.append("dinleyici sahnesi ve sanatçı topluluğu bağı")

    if cam_val >= 0.85 and tex_val >= 0.30 and len(out) < 4:
        c_key = f"{meta_list[cand_idx].get('key') or ''} {meta_list[cand_idx].get('scale') or ''}".strip()
        out.append(f"armonik ton uyumu ({c_key})" if c_key else "armonik ton uyumu")

    if era_val >= 0.72 and tex_val >= 0.25 and len(out) < 4:
        out.append("yakın dönem ve prodüksiyon dokusu")
    return out[:4] or (meta_list[cand_idx].get("sig_traits", [])[:2]) or ["genel ses dokusu benzer"]


def _graph_ppr_vector(seed_idx: int, seed_artist_override: str = "") -> tuple[np.ndarray, bool]:
    """Runs fast sparse Personalized PageRank over CSR artist_relations in <2 ms."""
    out = np.zeros(_N_TRACKS, dtype=np.float32)
    if _GRAPH_V == 0:
        return out, False

    p0_nodes: dict[int, float] = {}
    u0 = int(_TRACK_GRAPH_IDX[seed_idx])
    if u0 >= 0:
        p0_nodes[u0] = 1.0
    if seed_artist_override:
        ov_fold = _tr_fold(seed_artist_override)
        for oi in _BY_ARTIST_IDX.get(ov_fold, ()):
            uo = int(_TRACK_GRAPH_IDX[oi])
            if uo >= 0 and uo not in p0_nodes:
                p0_nodes[uo] = 0.65 if u0 >= 0 else 1.0
    if not p0_nodes:
        return out, False

    tot_w = sum(p0_nodes.values()) or 1.0
    p0 = {u: w / tot_w for u, w in p0_nodes.items()}
    scores = dict(p0)
    alpha = 0.28
    fwd_share = 0.75
    rev_share = 0.25

    for _ in range(12):
        nxt: dict[int, float] = {u: alpha * w for u, w in p0.items()}
        for u, sc in scores.items():
            if sc <= 1e-5:
                continue
            fs, fe = int(_GRAPH_FWD_PTR[u]), int(_GRAPH_FWD_PTR[u + 1])
            f_deg = fe - fs
            if f_deg > 0:
                pf = (1.0 - alpha) * fwd_share * sc / f_deg
                for k in range(fs, fe):
                    nb = int(_GRAPH_FWD_IDX[k])
                    nxt[nb] = nxt.get(nb, 0.0) + pf
            else:
                for su, sw in p0.items():
                    nxt[su] = nxt.get(su, 0.0) + (1.0 - alpha) * fwd_share * sc * sw

            rs, re_ = int(_GRAPH_REV_PTR[u]), int(_GRAPH_REV_PTR[u + 1])
            r_deg = re_ - rs
            if r_deg > 0:
                pr = (1.0 - alpha) * rev_share * sc / r_deg
                for k in range(rs, re_):
                    nb = int(_GRAPH_REV_IDX[k])
                    nxt[nb] = nxt.get(nb, 0.0) + pr
            else:
                for su, sw in p0.items():
                    nxt[su] = nxt.get(su, 0.0) + (1.0 - alpha) * rev_share * sc * sw
        scores = nxt

    node_vec = np.zeros(_GRAPH_V, dtype=np.float32)
    max_sc = max(scores.values()) if scores else 1.0
    if max_sc <= 0.0:
        return out, False
    for u, sc in scores.items():
        if u not in p0 and sc > 1e-6:
            node_vec[u] = float(np.sqrt(min(1.0, sc / max_sc)))

    has_node = _TRACK_GRAPH_IDX >= 0
    out[has_node] = node_vec[_TRACK_GRAPH_IDX[has_node]]
    has_behav = int((out > 0.0).sum()) >= 20
    return out, has_behav


def recommend_compact(seed_idx: int, params: Params, seed_artist_override: str = ""):
    mode_name = params.mode_name or ("adventurous" if params.gamma == 0.0 and params.delta == 0.0 else "balanced")
    N = _N_TRACKS
    s_nf = int(_NOISE_FLAGS[seed_idx])
    seed_artist = _ART_FOLD[seed_idx]
    override_artist = _tr_fold(seed_artist_override or "")
    seed_reg = "tr" if _REG_CODE[seed_idx] == 1 else "global"
    target_reg = params.force_region or (seed_reg if params.strict_region and seed_reg in ("tr", "global") else None)

    mask = np.ones(N, dtype=bool)
    mask[seed_idx] = False
    mask &= (_ART_GROUP_ID != _ART_GROUP_ID[seed_idx])
    if override_artist and override_artist != seed_artist and override_artist in _BY_ARTIST_IDX:
        for oi in _BY_ARTIST_IDX[override_artist]:
            mask[oi] = False
    s_kh = _CAT_KEY_HASH[seed_idx]
    if (mask & (_CAT_KEY_HASH == s_kh)).any():
        seed_tf = _TIT_FOLD[seed_idx]
        for ci in np.flatnonzero(_CAT_KEY_HASH == s_kh):
            if _TIT_FOLD[ci] == seed_tf:
                mask[ci] = False
    if target_reg:
        mask &= (_REG_CODE == (1 if target_reg == "tr" else 0))
    if (s_nf & _NOISE_JUNK_BIT) == 0:
        mask &= ((_NOISE_FLAGS & _NOISE_JUNK_BIT) == 0)

    if not mask.any():
        return []

    # Smart Live / Remix / Vocal-vs-Instrumental cover guard
    nf_pen = np.ones(N, dtype=np.float32)
    if (s_nf & _NOISE_LIVE_BIT) == 0:
        nf_pen[(_NOISE_FLAGS & _NOISE_LIVE_BIT) != 0] *= 0.45
    if (s_nf & _NOISE_REMIX_BIT) == 0:
        nf_pen[(_NOISE_FLAGS & _NOISE_REMIX_BIT) != 0] *= 0.50
    seed_is_vocal = float(matrices["F"][seed_idx, F_IDX["instrumental"]]) < 0.30 and not _INST_COVER_RX.search(_TITLES[seed_idx])
    if seed_is_vocal:
        nf_pen[matrices["F"][:, F_IDX["instrumental"]] >= 0.78] *= 0.50

    # 1. Deep Timbre & Vocal Texture (mode-weighted EffNet 1280d + CLAP 512d + vocal register coherence)
    clap_sim = _chunk_dot(matrices["C"], _mat_row("C", seed_idx), scales=_ROW_SCALES.get("C"))
    if matrices["has_effnet"][seed_idx] and matrices["has_effnet"].any():
        eff_sim = _chunk_dot(matrices["E"], _mat_row("E", seed_idx), scales=_ROW_SCALES.get("E"))
        if mode_name == "adventurous":
            tex_mix = 0.22 * eff_sim + 0.78 * clap_sim
            incoh_pen = 0.12 * np.maximum(0.0, 0.12 - clap_sim)
        elif mode_name == "emotional":
            tex_mix = 0.34 * eff_sim + 0.66 * clap_sim
            incoh_pen = 0.20 * np.maximum(0.0, 0.14 - np.minimum(eff_sim, clap_sim))
        else:
            tex_mix = 0.56 * eff_sim + 0.44 * clap_sim
            incoh_pen = 0.28 * np.maximum(0.0, 0.14 - np.minimum(eff_sim, clap_sim))
        tex_sim = np.where(matrices["has_effnet"], tex_mix - incoh_pen, clap_sim)
    else:
        tex_sim = clap_sim

    seed_vg = float(matrices["vocal_gender"][seed_idx])
    if abs(seed_vg) >= 0.22:
        vg_clash = np.maximum(0.0, -(seed_vg * matrices["vocal_gender"]) - 0.04)
        vg_w = 0.10 if mode_name == "adventurous" else 0.16
        tex_sim = tex_sim - vg_w * vg_clash

    # 2. Genre & Style Family similarity (relaxed in adventurous mode to allow cross-genre bridges)
    raw_genre_sim = 0.54 * _chunk_dot(matrices["G"], _mat_row("G", seed_idx), scales=_ROW_SCALES.get("G")) + 0.46 * (matrices["P"] @ matrices["P"][seed_idx])
    genre_sim = raw_genre_sim if mode_name == "adventurous" else _style_family_compatibility_compact(seed_idx, raw_genre_sim)

    # 3. Scene-relative Sonic & Arrangement Signature ('Şarkının İmzası')
    elig_other = np.arange(N) != seed_idx
    n_nb = int(max(60, 0.10 * elig_other.sum()))
    gs = np.where(elig_other, genre_sim, -np.inf)
    nb = np.argsort(-gs)[:n_nb]
    S = matrices["S"]
    mu_l = S[nb].mean(axis=0)
    sd_l = S[nb].std(axis=0) + 0.03

    local_seed_z = np.clip((S[seed_idx] - mu_l) / sd_l, -2.5, 3.5)
    seed_sig_z = (1.0 - SCENE_SIG_BLEND) * matrices["S_z"][seed_idx] + SCENE_SIG_BLEND * local_seed_z
    seed_sig_w = _compute_sig_weights(seed_sig_z)

    local_cz = np.clip((S - mu_l[None, :]) / sd_l[None, :], -2.5, 3.5)
    cz_sig = (1.0 - SCENE_SIG_BLEND) * matrices["S_z"] + SCENE_SIG_BLEND * local_cz

    deficit = np.maximum(0.0, np.minimum(seed_sig_z[None, :], 1.65) - cz_sig) / 1.85
    trait_sim = np.clip(1.0 - deficit, 0.0, 1.0)
    base_match = trait_sim @ seed_sig_w
    missing_hook = (seed_sig_z[None, :] >= 0.58) & (cz_sig < -0.10)
    miss_pen = np.where(missing_hook, np.clip((seed_sig_z[None, :] - cz_sig) / 1.6, 0.0, 1.5), 0.0) @ seed_sig_w

    salient_hooks = seed_sig_z >= 0.55
    if salient_hooks.sum() >= 3 and salient_hooks[2]:
        salient_hooks = salient_hooks.copy()
        salient_hooks[2] = False
    has_conj = salient_hooks.sum() >= 2
    if has_conj:
        cand_has_hook = (cz_sig[:, salient_hooks] >= 0.25).astype(np.float32)
        hw = seed_sig_w[salient_hooks]
        hook_cov = (cand_has_hook @ hw) / (hw.sum() + 1e-9)
        conj_bonus = 0.42 * (hook_cov ** 1.7)
    else:
        conj_bonus = np.float32(0.0)

    absent_mask = (seed_sig_z < -0.20).astype(np.float32)
    clash_pen = 0.14 * (np.maximum(0.0, cz_sig - 1.15) * absent_mask[None, :]).mean(axis=1)
    sig_sim = base_match + conj_bonus - 1.45 * miss_pen - clash_pen

    # 4. Mode-aligned salience Audio DNA + octave BPM precision + mood polarity guard
    seed_z = matrices["Z"][seed_idx]
    seed_f = matrices["F"][seed_idx]
    priors = _mode_feature_priors(mode_name)
    raw_w = priors * (np.abs(seed_z) ** params.p + params.w_floor)
    seed_w = (raw_w / (raw_w.sum() + 1e-9)).astype(np.float32)

    D = _distances_compact(seed_f, seed_z, matrices["F"], matrices["Z"])
    if mode_name == "balanced":
        diff = np.abs(matrices["F"][:, BPM_I] - seed_f[BPM_I])
        octave = np.minimum(np.abs(diff - 1.0), np.abs(diff + 1.0)) + 0.10
        D[:, BPM_I] = np.clip(np.minimum(diff, octave) / 0.25, 0.0, 1.0)
    elif mode_name == "emotional":
        diff = np.abs(matrices["F"][:, BPM_I] - seed_f[BPM_I])
        octave = np.minimum(np.abs(diff - 1.0), np.abs(diff + 1.0)) + 0.14
        D[:, BPM_I] = np.clip(np.minimum(diff, octave) / 0.90, 0.0, 1.0)

    mood_pen = _mood_polarity_penalty_compact(seed_f, matrices["F"])
    mood_w = 0.36 if mode_name == "emotional" else (0.22 if mode_name == "adventurous" else 0.28)
    feat_sim = (1.0 - D @ seed_w) - mood_w * mood_pen

    # 5. Lyric Theme & Semantic Similarity (with fair neutral baseline for unindexed tracks)
    lyr_sim, has_seed_lyrics, s_max_theme = _lyric_similarity_compact(seed_idx)

    # 6. Era, Camelot Harmonic Key & Co-Listening Artist Graph (PPR)
    seed_year = float(matrices["years"][seed_idx] or 2018.0)
    sigma_yr = 5.5 if seed_year >= 2019.0 else 10.0
    era_sim = np.exp(-(((matrices["years"] - seed_year) / sigma_yr) ** 2)).astype(np.float32)
    cam_sim = _camelot_similarity(seed_idx) if params.omega > 0 else np.full(N, 0.50, dtype=np.float32)
    b_vec, has_behav = _graph_ppr_vector(seed_idx, seed_artist_override)

    # 7. Pool-independent, seed-specific absolute component similarities in [0, 1]
    abs_tex = (1.0 / (1.0 + np.exp(-6.2 * (np.clip(tex_sim, -1.0, 1.0) - 0.27)))).astype(np.float32)
    max_sig = 1.42 if has_conj else 1.00
    abs_sig = np.clip(sig_sim / max_sig, 0.0, 1.0).astype(np.float32)
    abs_dna = np.clip(feat_sim, 0.0, 1.0).astype(np.float32)
    abs_gen = np.clip(genre_sim, 0.0, 1.0).astype(np.float32)
    abs_lyr = np.clip((lyr_sim + 0.10) / 1.05, 0.0, 1.0).astype(np.float32)

    w_lyr = (params.eta * (1.0 + 1.35 * max(0.0, s_max_theme - 0.55))) if has_seed_lyrics else 0.0
    w_tex = params.alpha + (0.0 if has_seed_lyrics else 0.40 * params.eta)
    w_sig = params.zeta + (0.0 if has_seed_lyrics else 0.25 * params.eta)
    w_dna = params.beta + (0.0 if has_seed_lyrics else 0.35 * params.eta)
    w_gen = params.gamma
    w_era = params.delta
    w_cam = params.omega
    w_sum = max(1e-6, w_tex + w_sig + w_dna + w_gen + w_era + w_lyr + w_cam)

    core = (
        (w_tex / w_sum) * abs_tex
        + (w_sig / w_sum) * abs_sig
        + (w_dna / w_sum) * abs_dna
        + (w_gen / w_sum) * abs_gen
        + (w_era / w_sum) * (era_sim * np.clip(abs_dna / 0.75, 0.20, 1.0))
        + (w_lyr / w_sum) * abs_lyr
        + (w_cam / w_sum) * cam_sim
    )

    # Mode-calibrated specificity gates
    if mode_name == "adventurous":
        g_tex = np.clip(0.45 + 0.55 * (abs_tex / 0.52), 0.35, 1.0)
        g_dna = np.clip(0.40 + 0.60 * (abs_dna / 0.78), 0.30, 1.0)
        g_sig = np.float32(1.0)
        g_gen = np.float32(1.0)
    elif mode_name == "emotional":
        g_tex = np.clip(0.50 + 0.50 * (abs_tex / 0.52), 0.42, 1.0)
        g_dna = np.clip(0.42 + 0.58 * (abs_dna / 0.76), 0.35, 1.0)
        g_sig = np.clip(0.76 + 0.24 * (abs_sig / 0.72), 0.68, 1.0) if salient_hooks.any() else np.float32(1.0)
        g_gen = np.clip(1.0 - 0.35 * np.maximum(0.0, 0.20 - genre_sim), 0.85, 1.0)
    else:
        g_tex = np.clip(0.50 + 0.50 * (abs_tex / 0.55), 0.42, 1.0)
        g_dna = np.clip(0.45 + 0.55 * (abs_dna / 0.78), 0.35, 1.0)
        g_sig = np.clip(0.62 + 0.38 * (abs_sig / 0.72), 0.50, 1.0) if salient_hooks.any() else np.float32(1.0)
        g_gen = np.clip(1.0 - 1.15 * np.maximum(0.0, 0.25 - genre_sim), 0.45, 1.0) if params.gamma > 0 else np.float32(1.0)

    if abs(seed_vg) >= 0.18:
        opp = -np.sign(seed_vg) * matrices["vocal_gender"]
        vg_pen = 0.14 if mode_name == "adventurous" else 0.22
        g_vg = np.clip(1.0 - vg_pen * np.clip((opp - 0.18) / 0.45, 0.0, 1.0), 0.78, 1.0)
    else:
        g_vg = np.float32(1.0)

    if has_seed_lyrics and s_max_theme >= 0.55:
        lyr_floor = 0.74 if mode_name == "emotional" else 0.86
        g_lyr = np.where(matrices["has_lyrics"] & (lyr_sim < 0.10), lyr_floor, 1.0).astype(np.float32)
    else:
        g_lyr = np.float32(1.0)

    # Behavioral listener-graph coherence & mode-aware Deezer studio floor
    if has_behav and mode_name != "adventurous":
        if mode_name == "emotional":
            g_behav = np.where(b_vec > 0.0, 0.96 + 0.06 * b_vec, 0.92).astype(np.float32)
        elif mode_name == "safe":
            g_behav = np.where(b_vec > 0.0, 0.90 + 0.14 * b_vec, 0.72).astype(np.float32)
        else:
            g_behav = np.where(b_vec > 0.0, 0.90 + 0.14 * b_vec, 0.80).astype(np.float32)
    else:
        g_behav = np.float32(1.0)

    rp_f = _RANK_PCT.astype(np.float32)
    if mode_name != "adventurous":
        q_mult = np.where(
            (_RANK_PCT > 0) & (_RANK_PCT < 32) & (b_vec == 0.0),
            0.88 + 0.12 * (rp_f / 32.0),
            np.where(_RANK_PCT > 0, 0.985 + 0.025 * (rp_f / 255.0), 1.0),
        ).astype(np.float32)
    else:
        q_mult = np.float32(1.0)

    raw_gated = core * g_tex * g_dna * g_sig * g_gen * g_vg * g_lyr * g_behav * q_mult * nf_pen

    P_mat = matrices["P"]
    seed_top_p = int(np.argmax(P_mat[seed_idx]))
    cand_top_p = np.argmax(P_mat, axis=1)

    if mode_name == "adventurous":
        # Dampen predictable same-family clones and elevate cohesive cross-genre hidden gems
        same_fam_w = np.maximum(
            (cand_top_p == seed_top_p).astype(np.float32),
            np.clip((P_mat[:, seed_top_p] - 0.22) / 0.45, 0.0, 1.0),
        )
        subgenre_over = np.clip((raw_genre_sim - 0.35) / 0.55, 0.0, 1.0)
        novelty_mult = 1.0 - 0.22 * same_fam_w - 0.14 * subgenre_over
        bridge_ok = (abs_dna >= 0.65) & (abs_tex >= 0.38)
        bridge_bonus = np.where(
            (cand_top_p != seed_top_p) & bridge_ok,
            0.08 * np.minimum(abs_dna, abs_tex + 0.25),
            0.0,
        ).astype(np.float32)
        gem_bonus = np.where(_RANK_PCT > 0, 0.022 * (1.0 - rp_f / 255.0), 0.008).astype(np.float32)
        raw_gated = np.clip(raw_gated * novelty_mult + bridge_bonus + gem_bonus, 0.0, 1.0)
    elif mode_name == "balanced":
        bpm_tight = 1.0 - D[:, BPM_I]
        raw_gated = raw_gated * (0.93 + 0.08 * bpm_tight)

    # 8. Two-Stage Retrieval (pool >= 220) -> Stage 2 Pairwise Rerank -> MMR Diversity Pick
    stage1 = np.where(mask, raw_gated, -np.inf).astype(np.float32)
    pool_n = min(max(params.pool, params.k * 28, 224), int(mask.sum()))
    top_cand_idx = np.argsort(-stage1)[:pool_n]
    pool_order = [int(i) for i in top_cand_idx if mask[i]]
    if not pool_order:
        return []

    p_arr = np.asarray(pool_order, dtype=np.int32)
    s2 = np.clip(raw_gated[p_arr], 0.0, 1.0).astype(np.float32)
    for pos, ci in enumerate(pool_order):
        ctit = _TITLES[ci]
        if (s_nf & _NOISE_REMIX_BIT) == 0 and (_NOISE_FLAGS[ci] & _NOISE_REMIX_BIT) == 0 and _REMIX_RX.search(ctit):
            s2[pos] *= 0.50
        elif seed_is_vocal and _INST_COVER_RX.search(ctit):
            s2[pos] *= 0.22
    if mode_name == "balanced":
        s2 += 0.042 * (1.0 - D[p_arr, BPM_I])
        if has_behav:
            s2 += 0.090 * b_vec[p_arr]
        s2 += np.where(_RANK_PCT[p_arr] > 0, 0.018 * (rp_f[p_arr] / 255.0), 0.0)
    elif mode_name == "emotional":
        emo_closeness = 1.0 - (
            D[p_arr, F_IDX["valence"]] + D[p_arr, F_IDX["arousal"]] + D[p_arr, F_IDX["sad"]]
        ) / 3.0
        s2 += 0.065 * np.clip(emo_closeness, 0.0, 1.0)
        if has_seed_lyrics:
            s2 += 0.032 * np.maximum(0.0, lyr_sim[p_arr] - 0.45)
        if has_behav:
            s2 += 0.025 * b_vec[p_arr]
    elif mode_name == "local_scene":
        if has_behav:
            s2 += 0.085 * b_vec[p_arr]
        s2 += 0.035 * abs_sig[p_arr]
    elif mode_name == "safe":
        if has_behav:
            s2 += 0.110 * b_vec[p_arr]
        s2 += 0.030 * abs_gen[p_arr]

    rel_arr = np.clip(0.10 + 0.89 * (np.clip(s2, 0.0, 1.0) ** 0.82), 0.10, 0.99).astype(np.float32)
    rerank_order = np.argsort(-rel_arr)
    pool_order = [pool_order[r] for r in rerank_order]
    rel = rel_arr[rerank_order]

    chosen = []
    seen_artists: set[int] = set()
    seen_keys: set[int] = set()
    pool_E = _mat_rows("E", pool_order)
    pool_C = _mat_rows("C", pool_order)
    pool_P = P_mat[pool_order].astype(np.float32)
    pool_top_p = cand_top_p[pool_order]
    max_sim = np.zeros(len(pool_order), dtype=np.float32)
    fam_counts: dict[int, int] = {}
    region_counts: dict[str, int] = {"tr": 0, "global": 0}
    alive = np.ones(len(pool_order), dtype=bool)
    max_tr_balanced = max(2, params.k // 2)

    while len(chosen) < params.k and alive.any():
        mmr = params.lam * rel - (1.0 - params.lam) * max_sim
        if mode_name == "adventurous":
            for pj in range(len(pool_order)):
                if alive[pj]:
                    fc = fam_counts.get(int(pool_top_p[pj]), 0)
                    if fc > 0:
                        mmr[pj] -= 0.055 * fc
        elif mode_name == "balanced" and not params.strict_region and seed_reg == "tr":
            tr_seen = region_counts.get("tr", 0)
            if tr_seen >= max_tr_balanced:
                pen = 0.055 * (tr_seen - max_tr_balanced + 1)
                for pj in range(len(pool_order)):
                    if alive[pj] and _REG_CODE[pool_order[pj]] == 1:
                        mmr[pj] -= pen

        mmr[~alive] = -np.inf
        j = int(np.argmax(mmr))
        alive[j] = False
        idx = pool_order[j]
        art_gid = int(_ART_GROUP_ID[idx])
        tkey_h = int(_CAT_KEY_HASH[idx])
        if art_gid in seen_artists or tkey_h in seen_keys:
            continue
        seen_artists.add(art_gid)
        seen_keys.add(tkey_h)
        fam = int(pool_top_p[j])
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        reg = "tr" if _REG_CODE[idx] == 1 else "global"
        region_counts[reg] = region_counts.get(reg, 0) + 1

        reasons = _explain_compact(
            seed_idx, seed_sig_z, seed_sig_w, seed_w, cz_sig,
            idx, D[idx], float(era_sim[idx]), float(tex_sim[idx]), float(cam_sim[idx]),
            mode_name=mode_name, seed_top_p=seed_top_p, cand_top_p=fam,
            behav_val=float(b_vec[idx]),
        )
        cal_val = float(rel[j])
        chosen.append({
            "idx": idx,
            "score": cal_val,
            "match_percent": int(min(99, max(10, round(cal_val * 100)))),
            "tex_sim": float(tex_sim[idx]),
            "feat_sim": float(feat_sim[idx]),
            "sig_sim": float(sig_sim[idx]),
            "lyric_sim": float(lyr_sim[idx]),
            "genre_sim": float(genre_sim[idx]),
            "camelot_sim": float(cam_sim[idx]),
            "because": reasons,
        })
        if mode_name == "adventurous":
            sims = np.clip(0.55 * (pool_C @ pool_C[j]) + 0.45 * (pool_P @ pool_P[j]), 0.0, 1.0)
        elif matrices["has_effnet"][idx]:
            sims = np.clip(0.65 * (pool_E @ pool_E[j]) + 0.35 * (pool_C @ pool_C[j]), 0.0, 1.0)
        else:
            sims = np.clip(pool_C @ pool_C[j], 0.0, 1.0)
        max_sim = np.maximum(max_sim, sims)

    chosen.sort(key=lambda item: (item["match_percent"], item["score"], item["feat_sim"]), reverse=True)
    return chosen


HTML_PAGE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Suggestify — Müzikal Ruh ve Ses İkizleri</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #090a0f;
  --surface: #12141f;
  --surface-hover: #1b1e2e;
  --border: #23273a;
  --primary: #6366f1;
  --primary-glow: rgba(99, 102, 241, 0.25);
  --accent: #ec4899;
  --text: #f3f4f6;
  --text-muted: #9ca3af;
  --badge-bg: rgba(99, 102, 241, 0.12);
  --badge-border: rgba(99, 102, 241, 0.3);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Plus Jakarta Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-image: 
    radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
    radial-gradient(circle at 85% 80%, rgba(236, 72, 153, 0.08) 0%, transparent 40%);
}
header {
  padding: 2.5rem 1.5rem 1.5rem;
  text-align: center;
  border-bottom: 1px solid var(--border);
  background: rgba(18, 20, 31, 0.7);
  backdrop-filter: blur(12px);
}
.brand {
  font-size: 2.2rem;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #fff 0%, #a5b4fc 50%, #f472b6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}
.tagline { color: var(--text-muted); font-size: 0.95rem; margin-top: 0.4rem; }
main { max-width: 1000px; width: 100%; margin: 0 auto; padding: 2rem 1.5rem 4rem; flex: 1; }
.search-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.25rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  margin-bottom: 2rem;
}
.search-row { display: flex; gap: 0.75rem; position: relative; }
.autocomplete-list {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: #141724;
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-top: 6px;
  max-height: 280px;
  overflow-y: auto;
  z-index: 100;
  box-shadow: 0 12px 32px rgba(0,0,0,0.6);
  text-align: left;
}
.ac-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  cursor: pointer;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  transition: background 0.15s ease;
}
.ac-item:hover { background: #1f2335; }
.ac-thumb { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; }
.ac-info { flex: 1; min-width: 0; }
.ac-title { font-size: 0.9rem; font-weight: 600; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ac-artist { font-size: 0.8rem; color: var(--text-muted); }
input[type="text"] {
  flex: 1;
  background: #0d0f17;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.85rem 1.2rem;
  color: #fff;
  font-size: 1rem;
  outline: none;
  transition: all 0.2s;
}
input[type="text"]:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-glow);
}
button.btn-search {
  background: linear-gradient(135deg, var(--primary), #4f46e5);
  color: #fff;
  border: none;
  border-radius: 12px;
  padding: 0 1.8rem;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  transition: transform 0.1s, opacity 0.2s;
}
button.btn-search:hover { opacity: 0.95; }
button.btn-search:active { transform: scale(0.98); }
.modes-row {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
  flex-wrap: wrap;
  align-items: center;
}
.mode-label { font-size: 0.8rem; color: var(--text-muted); margin-right: 0.5rem; }
.mode-btn {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-muted);
  padding: 0.4rem 0.9rem;
  border-radius: 20px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}
.mode-btn.active {
  background: var(--badge-bg);
  border-color: var(--primary);
  color: #c7d2fe;
  font-weight: 600;
}
.quick-picks {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.8rem;
  flex-wrap: wrap;
  align-items: center;
}
.chip {
  background: #181b2a;
  border: 1px solid #282c40;
  color: #94a3b8;
  padding: 0.25rem 0.65rem;
  border-radius: 8px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
}
.chip:hover { background: #23283f; color: #fff; }

.seed-card {
  background: linear-gradient(135deg, rgba(30, 34, 53, 0.6), rgba(18, 20, 31, 0.8));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  display: flex;
  gap: 1.5rem;
  align-items: center;
}
.seed-cover {
  width: 100px;
  height: 100px;
  border-radius: 12px;
  object-fit: cover;
  background: #23273a;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}
.seed-info { flex: 1; }
.seed-title { font-size: 1.35rem; font-weight: 700; margin-bottom: 0.25rem; }
.seed-artist { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 0.6rem; }
.pill-group { display: flex; gap: 0.4rem; flex-wrap: wrap; }
.pill {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  font-size: 0.75rem;
  color: #cbd5e1;
}
.pill.sig { background: var(--badge-bg); border-color: var(--badge-border); color: #c7d2fe; }
.pill.lyric { background: rgba(236, 72, 153, 0.12); border-color: rgba(236, 72, 153, 0.3); color: #fbcfe8; }

.recs-heading {
  font-size: 1.15rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.rec-list { display: flex; flex-direction: column; gap: 0.75rem; }
.rec-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  display: flex;
  align-items: center;
  gap: 1.25rem;
  transition: all 0.2s;
}
.rec-card:hover {
  background: var(--surface-hover);
  border-color: #313752;
  transform: translateY(-1px);
}
.rec-rank { font-size: 1.1rem; font-weight: 700; color: var(--text-muted); width: 24px; }
.rec-cover { width: 56px; height: 56px; border-radius: 8px; object-fit: cover; background: #1e2235; }
.rec-main { flex: 1; }
.rec-title-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem; }
.rec-title { font-weight: 600; font-size: 0.98rem; }
.rec-year { color: var(--text-muted); font-size: 0.8rem; }
.rec-region-tag {
  font-size: 0.68rem;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  font-weight: 600;
  text-transform: uppercase;
}
.rec-region-tr { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
.rec-region-global { background: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
.rec-artist { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.4rem; }
.rec-why { font-size: 0.78rem; color: #94a3b8; display: flex; gap: 0.4rem; flex-wrap: wrap; }
.why-tag {
  background: rgba(255, 255, 255, 0.04);
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}
.rec-scores { text-align: right; }
.dna-match { font-size: 1.1rem; font-weight: 700; color: #38bdf8; }
.dna-sub { font-size: 0.7rem; color: var(--text-muted); }
.btn-play {
  background: #1e2238;
  border: 1px solid #333956;
  color: #fff;
  width: 38px;
  height: 38px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.85rem;
}
.btn-play:hover { background: var(--primary); border-color: var(--primary); }
.btn-play.playing { background: var(--accent); border-color: var(--accent); }

.player-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(18, 20, 31, 0.95);
  backdrop-filter: blur(16px);
  border-top: 1px solid var(--border);
  padding: 0.85rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  transform: translateY(100%);
  transition: transform 0.3s ease;
}
.player-bar.visible { transform: translateY(0); }
.player-info { display: flex; align-items: center; gap: 0.75rem; }
.player-text { font-size: 0.85rem; }
.player-track { font-weight: 600; color: #fff; }
.player-artist { color: var(--text-muted); font-size: 0.75rem; }
audio { outline: none; height: 32px; }

.spinner {
  display: none;
  width: 36px;
  height: 36px;
  border: 3px solid rgba(255,255,255,0.1);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 3rem auto;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<header>
  <div class="brand">✨ Suggestify</div>
  <div class="tagline">Müzikal Ruh ve Ses İkizleri — __TRACK_COUNT__ Şarkılık Duygu Odaklı Öneri Motoru</div>
</header>

<main>
  <div class="search-box">
    <div class="search-row">
      <input type="text" id="queryInput" placeholder="Bir şarkı veya sanatçı yazın... (örn: Dedublüman, Mor ve Ötesi, Adele)" autocomplete="off">
      <button class="btn-search" onclick="doSearch()">Keşfet</button>
      <div id="autocompleteList" class="autocomplete-list" style="display:none;"></div>
    </div>
    <div class="modes-row">
      <span class="mode-label">Keşif Modu:</span>
      <button class="mode-btn active" data-mode="emotional" onclick="setMode('emotional', this)">🌟 Duygu & His (Sınır Tanımayan)</button>
      <button class="mode-btn" data-mode="balanced" onclick="setMode('balanced', this)">⚖️ Dengeli Hibrit</button>
      <button class="mode-btn" data-mode="local_scene" onclick="setMode('local_scene', this)">🇹🇷 Sadece Yerel Sahne</button>
      <button class="mode-btn" data-mode="adventurous" onclick="setMode('adventurous', this)">🧭 Macera (Sürpriz Tür)</button>
    </div>
    <div class="quick-picks">
      <span class="mode-label">Hızlı Örnekler:</span>
      <span class="chip" onclick="quick('Dedublüman Günü Gelir')">Dedublüman - Günü Gelir</span>
      <span class="chip" onclick="quick('Wegh Geri Ver')">Wegh - Geri Ver</span>
      <span class="chip" onclick="quick('Adele Someone Like You')">Adele - Someone Like You</span>
      <span class="chip" onclick="quick('Altın Gün Yüce Dağ Başında')">Altın Gün - Yüce Dağ Başında</span>
      <span class="chip" onclick="quick('Mor ve Ötesi Bir Derdim Var')">Mor ve Ötesi - Bir Derdim Var</span>
      <span class="chip" onclick="quick('M83 Midnight City')">M83 - Midnight City</span>
      <span class="chip" onclick="quick('Jakuzi Koca Bir Saçmalık')">Jakuzi - Koca Bir Saçmalık</span>
    </div>
  </div>

  <div id="spinner" class="spinner"></div>

  <div id="resultsArea" style="display:none;">
    <div id="seedContainer"></div>
    <div class="recs-heading">
      <span>Önerilen Ses İkizleri</span>
      <span id="recsCount" style="font-size:0.85rem; color:var(--text-muted); font-weight:400;"></span>
    </div>
    <div class="rec-list" id="recList"></div>
  </div>
</main>

<div class="player-bar" id="playerBar">
  <div class="player-info">
    <div class="btn-play playing" id="playerBtn" onclick="toggleAudio()">❚❚</div>
    <div class="player-text">
      <div class="player-track" id="playerTitle">Şarkı Adı</div>
      <div class="player-artist" id="playerArtist">Sanatçı</div>
    </div>
  </div>
  <audio id="audioElem" controls></audio>
</div>

<script>
let currentMode = 'emotional';
const audioElem = document.getElementById('audioElem');
const playerBar = document.getElementById('playerBar');
const queryInput = document.getElementById('queryInput');
const acList = document.getElementById('autocompleteList');
let currentPlayBtn = null;
let searchTimeout = null;

queryInput.addEventListener('input', () => {
  const q = queryInput.value.trim();
  if (searchTimeout) clearTimeout(searchTimeout);
  if (q.length < 2) {
    acList.style.display = 'none';
    return;
  }
  searchTimeout = setTimeout(async () => {
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=8`);
      const items = await res.json();
      if (!Array.isArray(items) || items.length === 0) {
        acList.style.display = 'none';
        return;
      }
      acList.innerHTML = items.map(item => `
        <div class="ac-item" onclick="selectTrack('${(item.artist_name + ' - ' + item.title).replace(/'/g, "\\'")}')">
          <img src="${item.cover_url || 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=80&auto=format&fit=crop&q=80'}" class="ac-thumb">
          <div class="ac-info">
            <div class="ac-title">${item.title}</div>
            <div class="ac-artist">${item.artist_name}</div>
          </div>
        </div>
      `).join('');
      acList.style.display = 'block';
    } catch(e) {
      acList.style.display = 'none';
    }
  }, 160);
});

function selectTrack(str) {
  queryInput.value = str;
  acList.style.display = 'none';
  doSearch();
}

document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-box')) {
    acList.style.display = 'none';
  }
});

function setMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const q = queryInput.value.trim();
  if (q) doSearch();
}

function quick(q) {
  queryInput.value = q;
  acList.style.display = 'none';
  doSearch();
}

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    acList.style.display = 'none';
    doSearch();
  }
});

async function doSearch() {
  acList.style.display = 'none';
  const q = queryInput.value.trim();
  if (!q) return;

  const spinner = document.getElementById('spinner');
  const results = document.getElementById('resultsArea');
  spinner.style.display = 'block';
  results.style.display = 'none';

  try {
    const res = await fetch(`/api/recommend?query=${encodeURIComponent(q)}&mode=${currentMode}&k=8`);
    const data = await res.json();
    spinner.style.display = 'none';
    if (data.error) {
      alert("Hata: " + data.error);
      return;
    }
    renderResults(data);
    results.style.display = 'block';
  } catch (err) {
    spinner.style.display = 'none';
    alert("Bağlantı hatası: " + err);
  }
}

function escapeHtml(val) {
  return String(val ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function safeMediaUrl(url, fallback) {
  if (typeof url === 'string' && /^https?:\\/\\//i.test(url)) return url;
  return fallback || '';
}

const TRACK_PLAY_ITEMS = [];

function playTrackIndex(idx, btn) {
  const item = TRACK_PLAY_ITEMS[idx];
  if (!item || !item.preview) return;
  playAudio(item.preview, item.title, item.artist, btn);
}

function renderResults(data) {
  TRACK_PLAY_ITEMS.length = 0;
  const seed = data.seed;
  const recs = data.recommendations || [];

  const seedCover = safeMediaUrl(
    seed.cover_url,
    'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=200&auto=format&fit=crop&q=80'
  );
  const seedPreview = safeMediaUrl(seed.preview, '');
  let seedPlayIdx = -1;
  if (seedPreview) {
    seedPlayIdx = TRACK_PLAY_ITEMS.push({
      preview: seedPreview,
      title: String(seed.title || ''),
      artist: String(seed.artist_name || ''),
    }) - 1;
  }

  document.getElementById('seedContainer').innerHTML = `
    <div class="seed-card">
      <img src="${escapeHtml(seedCover)}" class="seed-cover" alt="cover">
      <div class="seed-info">
        <div class="seed-title">${escapeHtml(seed.title)}</div>
        <div class="seed-artist">${escapeHtml(seed.artist_name)} · ${escapeHtml(seed.year || '')} (${escapeHtml(seed.region ? String(seed.region).toUpperCase() : 'TR')})</div>
        <div class="pill-group">
          <span class="pill">Tempo: ${escapeHtml(seed.bpm)} BPM</span>
          <span class="pill">${escapeHtml(seed.key)}</span>
          ${seed.lyric_dominant ? `<span class="pill lyric">Söz: ${escapeHtml(seed.lyric_label)}</span>` : ''}
          ${(seed.signatures || []).map(s => `<span class="pill sig">${escapeHtml(s)}</span>`).join('')}
        </div>
      </div>
      ${seedPlayIdx >= 0 ? `<button class="btn-play" onclick="playTrackIndex(${seedPlayIdx}, this)">▶</button>` : ''}
    </div>
  `;

  document.getElementById('recsCount').innerText = `${recs.length} Şarkı Eşleşti`;
  const list = document.getElementById('recList');
  const getMatchPct = (item) => {
    if (typeof item?.match_percent === 'number' && !Number.isNaN(item.match_percent)) {
      return Math.min(99, Math.max(10, Math.round(item.match_percent)));
    }
    return Math.min(99, Math.max(10, Math.round(Number(item?.dna || 0) * 100)));
  };
  const sortedRecs = [...(recs || [])].sort((a, b) => {
    const pctDiff = getMatchPct(b) - getMatchPct(a);
    if (pctDiff !== 0) return pctDiff;
    const scoreDiff = Number(b?.score || 0) - Number(a?.score || 0);
    if (scoreDiff !== 0) return scoreDiff;
    return Number(b?.dna || 0) - Number(a?.dna || 0);
  });
  list.innerHTML = sortedRecs.map((r, i) => {
    const cover = safeMediaUrl(
      r.cover_url,
      'https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=120&auto=format&fit=crop&q=80'
    );
    const preview = safeMediaUrl(r.preview, '');
    let recPlayIdx = -1;
    if (preview) {
      recPlayIdx = TRACK_PLAY_ITEMS.push({
        preview,
        title: String(r.title || ''),
        artist: String(r.artist_name || ''),
      }) - 1;
    }
    const regClass = r.region === 'tr' ? 'rec-region-tr' : 'rec-region-global';
    const pct = getMatchPct(r);
    return `
      <div class="rec-card">
        <div class="rec-rank">#${i+1}</div>
        <img src="${escapeHtml(cover)}" class="rec-cover" alt="cover">
        <div class="rec-main">
          <div class="rec-title-row">
            <span class="rec-title">${escapeHtml(r.title)}</span>
            <span class="rec-year">${escapeHtml(r.year || '')}</span>
            <span class="rec-region-tag ${regClass}">${escapeHtml(r.region)}</span>
          </div>
          <div class="rec-artist">${escapeHtml(r.artist_name)}</div>
          <div class="rec-why">
            ${(r.because || []).map(b => `<span class="why-tag">${escapeHtml(b)}</span>`).join('')}
          </div>
        </div>
        <div class="rec-scores">
          <div class="dna-match">%${pct}</div>
          <div class="dna-sub">Ses Uyumu</div>
        </div>
        ${recPlayIdx >= 0 ? `<button class="btn-play" onclick="playTrackIndex(${recPlayIdx}, this)">▶</button>` : ''}
      </div>
    `;
  }).join('');
}

function playAudio(url, title, artist, btn) {
  if (currentPlayBtn) currentPlayBtn.classList.remove('playing');
  audioElem.src = url;
  audioElem.play();
  playerBar.classList.add('visible');
  document.getElementById('playerTitle').innerText = title;
  document.getElementById('playerArtist').innerText = artist;
  document.getElementById('playerBtn').innerText = '❚❚';
  btn.classList.add('playing');
  currentPlayBtn = btn;
}

function toggleAudio() {
  if (audioElem.paused) {
    audioElem.play();
    document.getElementById('playerBtn').innerText = '❚❚';
  } else {
    audioElem.pause();
    document.getElementById('playerBtn').innerText = '▶';
  }
}
</script>

</body>
</html>
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"


class RequestHandler(SimpleHTTPRequestHandler):
    def address_string(self):
        return str(self.client_address[0])

    def _send_cors_headers(self, code=200):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "close")
        self.close_connection = True

    def do_OPTIONS(self):
        self._send_cors_headers(204)
        self.end_headers()
        self.wfile.flush()

    def do_GET(self):
        print(f"--> [HTTP GET] {self.path}", flush=True)
        parsed = urlparse(self.path)
        if parsed.path == "/api/recommend":
            self.handle_recommend(parse_qs(parsed.query))
        elif parsed.path == "/api/search":
            self.handle_search(parse_qs(parsed.query))
        elif parsed.path in ("/", "/index.html"):
            track_count_str = f"{len(meta_list):,}".replace(",", ".")
            payload = HTML_PAGE.replace("__TRACK_COUNT__", track_count_str).encode("utf-8")
            self._send_cors_headers(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif parsed.path == "/health":
            self._send_json({"status": "healthy", "tracks": len(meta_list)})
        else:
            self._send_cors_headers(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def _score_search_candidate(self, q_fold: str, art_q: str, tit_q: str, a_fold: str, t_fold: str, m: dict, prefer_tr: bool) -> int:
        score = 0
        if art_q and tit_q:
            if a_fold == art_q and t_fold == tit_q:
                score += 220
            elif art_q in a_fold and tit_q in t_fold:
                score += 150
        if a_fold == q_fold:
            score += 140
        elif a_fold.startswith(q_fold):
            score += 95
        if t_fold == q_fold:
            score += 135
        elif t_fold.startswith(q_fold) and a_fold != q_fold:
            score += 85
        elif a_fold != q_fold and (f" {q_fold}" in f" {a_fold}" or f" {q_fold}" in f" {t_fold}"):
            score += 50
        elif q_fold in a_fold or q_fold in t_fold:
            score += 20
        if m.get("region") == "tr":
            score += 120 if prefer_tr else 85
        year = int(m.get("year") or 2000)
        score += min(15, max(0, year - 2010))
        return score

    def handle_search(self, params):
        q_list = params.get("q") or params.get("query") or [""]
        raw_q = q_list[0].strip()
        clean_q = raw_q.replace("—", " - ").replace("–", " - ").replace("_", " ").strip()
        q = _tr_fold(clean_q)
        if not q:
            self._send_json([])
            return

        mode_name = params.get("mode", [""])[0]
        prefer_tr = mode_name == "local_scene" or params.get("region", [""])[0] == "tr"
        try:
            raw_limit = int(params.get("limit", [10])[0])
        except (TypeError, ValueError):
            self._send_json({"error": "Geçersiz 'limit' parametresi"}, 400)
            return
        limit = max(1, min(50, raw_limit))

        art_q, tit_q = ("", "")
        if " - " in clean_q:
            p0, p1 = clean_q.split(" - ", 1)
            art_q, tit_q = _tr_fold(p0), _tr_fold(p1)

        q_comp = q.replace(" ", "")
        has_space = " " in q
        q_tokens = q.split() if has_space else ()
        scored: list[tuple[int, dict]] = []
        seen = set()

        for idx in range(_N_TRACKS):
            a = _ART_FOLD[idx]
            t = _TIT_FOLD[idx]
            matched = (q in t or q in a)
            if not matched and has_space:
                matched = (art_q and tit_q and art_q in a and tit_q in t) or (
                    all(tok in a or tok in t for tok in q_tokens) and q in f"{a} {t}"
                )
            if not matched and len(q_comp) >= 4:
                matched = q_comp in f"{a}{t}".replace(" ", "")
            if not matched:
                continue
            key = f"{a}::{t}"
            if key in seen:
                continue
            seen.add(key)
            m = meta_list[idx]
            sc = self._score_search_candidate(q, art_q, tit_q, a, t, m, prefer_tr)
            # Clean studio boost over live/remix duplicates in search dropdown
            if (int(_NOISE_FLAGS[idx]) & 7) == 0:
                sc += 18
            tid = int(matrices["ids"][idx])
            cached = _cache_media_get(tid)
            cov = m.get("cover_url") or (cached[0] if cached else None)
            prev = cached[1] if cached else None
            scored.append((sc, {
                "id": tid,
                "title": _TITLES[idx],
                "artist_name": _ARTISTS[idx],
                "genre": self._extract_genre(m, idx),
                "preview": prev,
                "cover_url": cov,
                "year": m.get("year"),
                "region": m.get("region"),
            }))

        # Always blend online Deezer/iTunes matches for queries >= 2 chars when cache/live helps surface missing or new artists
        if len(q) >= 2:
            online = search_online_tracks(clean_q, limit=max(5, limit), prefer_tr=prefer_tr)
            for idx_on, item in enumerate(online):
                a_on = _tr_fold(item.get("artist_name") or "")
                t_on = _tr_fold(item.get("title") or "")
                key = f"{a_on}::{t_on}"
                if key in seen:
                    continue
                seen.add(key)
                sc = self._score_search_candidate(q, art_q, tit_q, a_on, t_on, item, prefer_tr)
                # Slight boost for online direct popularity when local has no exact artist match
                sc += max(5, 35 - idx_on * 6)
                scored.append((sc, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        matches = [pair[1] for pair in scored[:limit]]
        self._send_json(matches)

    def _extract_genre(self, m, idx: int | None = None):
        if m.get("genre"):
            return m["genre"]
        if idx is not None and 0 <= idx < len(matrices["P"]):
            p_row = matrices["P"][idx]
            top_p = int(np.argmax(p_row))
            if float(p_row[top_p]) >= 0.25:
                if top_p in (P_FAMILY["pop_ballad"], P_FAMILY["pop_mainstream"]):
                    return "Türkçe Pop" if m.get("region") == "tr" else "Pop"
                if top_p in _P_FAMILY_LABELS:
                    return _P_FAMILY_LABELS[top_p]
        sig = m.get("sig_traits") or []
        joined = " ".join(sig).lower()
        if "rock" in joined or "gitar" in joined:
            return "Rock / Alternatif"
        if "rap" in joined or "trap" in joined or "808" in joined:
            return "Rap / Hip-Hop"
        if "akustik" in joined or "piyano" in joined:
            return "Akustik / Indie"
        if "elektronik" in joined or "synth" in joined:
            return "Elektronik / Pop"
        if "yaylılar" in joined or "senfonik" in joined:
            return "Klasik / Pop"
        if m.get("region") == "tr":
            return "Türkçe Pop"
        return "Pop"

    def handle_recommend(self, params):
        id_list = params.get("id") or params.get("track_id")
        q_list = params.get("query") or params.get("q") or [""]
        q = q_list[0].strip()
        mode_name = params.get("mode", ["emotional"])[0]
        try:
            raw_k = int(params.get("k", [8])[0])
        except (TypeError, ValueError):
            self._send_json({"error": "Geçersiz 'k' parametresi"}, 400)
            return
        k = max(1, min(50, raw_k))

        if not q and not (id_list and id_list[0].strip()):
            self._send_json({"error": "Sorgu belirtilmedi"}, 400)
            return

        try:
            prefer_tr = mode_name == "local_scene"
            seed_idx = None
            seed_meta = None
            if id_list and id_list[0].strip().isdigit() and int(id_list[0].strip()) in by_id:
                seed_idx, seed_meta = by_id[int(id_list[0].strip())]
            else:
                seed_idx, seed_meta = self._resolve(q or (id_list[0] if id_list else ""), prefer_tr=prefer_tr)

            if seed_idx is None or seed_meta is None:
                self._send_json({"error": f"'{q}' için eşleşen parça bulunamadı"}, 404)
                return

            params_obj = replace(MODES.get(mode_name, MODES["emotional"]), k=k)
            recs = recommend_compact(seed_idx, params_obj, seed_artist_override=seed_meta.get("artist_name", ""))

            # Hydrate cover_url and preview for seed + recommended tracks concurrently
            hyd_inputs = [(int(seed_meta["id"]), seed_meta.get("cover_url"), seed_meta.get("preview"), seed_meta.get("artist_name", ""), seed_meta.get("title", ""))]
            for r in recs:
                rm = meta_list[r["idx"]]
                hyd_inputs.append((int(rm["id"]), rm.get("cover_url"), rm.get("preview"), rm.get("artist_name", ""), rm.get("title", "")))
            media_map = hydrate_media_batch(hyd_inputs)

            seed_cov, seed_prev = media_map.get(int(seed_meta["id"]), (seed_meta.get("cover_url"), seed_meta.get("preview")))
            seed_f = matrices["F"][seed_idx]
            seed_bpm = seed_meta.get("bpm") or (2.0 ** float(seed_f[BPM_I]))
            dom_theme = seed_meta.get("lyric_dominant") or "neutral"

            seed_data = {
                "id": seed_meta["id"],
                "title": seed_meta["title"],
                "artist_name": seed_meta["artist_name"],
                "year": seed_meta.get("year"),
                "region": seed_meta.get("region"),
                "cover_url": seed_cov,
                "preview": seed_prev,
                "bpm": round(float(seed_bpm)),
                "key": f"{seed_meta.get('key') or 'C'} {seed_meta.get('scale') or 'major'}",
                "lyric_dominant": dom_theme,
                "lyric_label": THEME_LABELS.get(dom_theme, dom_theme),
                "signatures": seed_meta.get("sig_traits", []),
                "is_proxy_analysis": bool(seed_meta.get("is_proxy_analysis", False)),
                "resolved_from": seed_meta.get("resolved_from"),
                "dnaFeatures": {
                    "energy": round(float(seed_f[F_IDX["arousal"]]), 3),
                    "valence": round(float(seed_f[F_IDX["valence"]]), 3),
                    "danceability": round(float(seed_f[F_IDX["danceability"]]), 3),
                    "acousticness": round(float(seed_f[F_IDX["acoustic"]]), 3),
                    "happy": round(float(seed_f[F_IDX["happy"]]), 3),
                    "sad": round(float(seed_f[F_IDX["sad"]]), 3),
                    "aggressive": round(float(seed_f[F_IDX["aggressive"]]), 3),
                },
            }

            rec_list = []
            for r in recs:
                idx = r["idx"]
                m = meta_list[idx]
                cf = matrices["F"][idx]
                c_bpm = m.get("bpm") or (2.0 ** float(cf[BPM_I]))
                c_cov, c_prev = media_map.get(int(m["id"]), (m.get("cover_url"), m.get("preview")))
                tempo_norm = float(np.clip((float(c_bpm) - 60.0) / 110.0, 0.05, 0.99))
                lyr_norm = float(np.clip(0.5 + 0.5 * r["lyric_sim"], 0.05, 0.99)) if matrices["has_lyrics"][idx] else 0.45
                rec_list.append({
                    "id": m["id"],
                    "artist_name": m["artist_name"],
                    "title": m["title"],
                    "year": m.get("year"),
                    "region": m.get("region"),
                    "genre": self._extract_genre(m, idx),
                    "bpm": round(float(c_bpm)),
                    "cover_url": c_cov,
                    "preview": c_prev,
                    "because": r["because"],
                    "match_percent": int(r["match_percent"]),
                    "score": round(r["score"], 3),
                    "dna": round(r["feat_sim"], 2),
                    "sig": round(r["sig_sim"], 2),
                    "tex_sim": round(r["tex_sim"], 3),
                    "lyric_match": round(r["lyric_sim"], 3),
                    "camelot_compat": round(r.get("camelot_sim", 0.5), 2),
                    "radar": {
                        "energy": round(float(np.clip(cf[F_IDX["arousal"]], 0.05, 0.99)), 3),
                        "valence": round(float(np.clip(cf[F_IDX["valence"]], 0.05, 0.99)), 3),
                        "acousticness": round(float(np.clip(cf[F_IDX["acoustic"]], 0.05, 0.99)), 3),
                        "danceability": round(float(np.clip(cf[F_IDX["danceability"]], 0.05, 0.99)), 3),
                        "tempo": round(tempo_norm, 3),
                        "lyrics": round(lyr_norm, 3),
                    },
                })

            rec_list.sort(
                key=lambda item: (
                    int(item.get("match_percent") or 0),
                    float(item.get("score") or 0.0),
                    float(item.get("dna") or 0.0),
                ),
                reverse=True,
            )

            self._send_json({"seed": seed_data, "recommendations": rec_list})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _make_seed_proxy(self, best_m: dict, online_item: dict) -> dict:
        seed_copy = dict(best_m)
        seed_copy["id"] = online_item["id"]
        seed_copy["title"] = online_item["title"]
        seed_copy["artist_name"] = online_item["artist_name"]
        seed_copy["preview"] = online_item.get("preview") or best_m.get("preview")
        seed_copy["is_proxy_analysis"] = True
        seed_copy["resolved_from"] = f"{best_m.get('artist_name')} - {best_m.get('title')}"
        if online_item.get("cover_url"):
            seed_copy["cover_url"] = online_item["cover_url"]
        if online_item.get("year"):
            seed_copy["year"] = online_item["year"]
        return seed_copy

    def _resolve(self, q, prefer_tr: bool = False):
        clean_q = q.replace("—", " - ").replace("–", " - ").replace("_", " ").strip()
        q_norm = _tr_fold(clean_q)
        if not q_norm:
            return None, None

        # 1. Exact catalog ID
        if q_norm.isdigit():
            tid = int(q_norm)
            if tid in by_id:
                return by_id[tid]

        # 2. Exact folded artist::title key in catalog
        if q_norm in by_key:
            return by_key[q_norm]

        # 3. Exact then substring 'Artist - Title' match in catalog
        if " - " in clean_q:
            parts = [p.strip() for p in clean_q.split(" - ", 1)]
            art_t = _tr_fold(parts[0])
            tit_t = _tr_fold(parts[1])
            exact_hits = []
            sub_hits = []
            for idx in range(_N_TRACKS):
                a = _ART_FOLD[idx]
                t = _TIT_FOLD[idx]
                if a == art_t and t == tit_t:
                    exact_hits.append((idx, meta_list[idx]))
                elif art_t and tit_t and ((art_t in a and tit_t in t) or (art_t in t and tit_t in a)):
                    sub_hits.append((idx, meta_list[idx]))
            for pool in (exact_hits, sub_hits):
                if pool:
                    return _pick_artist_rep_track(pool, prefer_tr=prefer_tr)

        # 4. Combined artist + title present in query (e.g., "Adele Someone Like You")
        combo_hits = []
        for idx in range(_N_TRACKS):
            a = _ART_FOLD[idx]
            t = _TIT_FOLD[idx]
            if a and a in q_norm and t and t in q_norm:
                score = (200 if _REG_CODE[idx] == 1 and prefer_tr else 0) + len(a) + len(t)
                combo_hits.append((score, idx))
        if combo_hits:
            combo_hits.sort(key=lambda x: x[0], reverse=True)
            s0 = combo_hits[0][0]
            best_combo = [(idx, meta_list[idx]) for sc, idx in combo_hits if s0 - sc <= 2][:6]
            return _pick_artist_rep_track(best_combo, prefer_tr=prefer_tr) if len(best_combo) > 1 else (combo_hits[0][1], meta_list[combo_hits[0][1]])

        # 4b. Exact catalog artist query (before global online search hijacks homonyms like "Motive")
        q_comp = q_norm.replace(" ", "")
        if q_norm in by_artist:
            return _pick_artist_rep_track(by_artist[q_norm], prefer_tr=prefer_tr)
        if len(q_comp) >= 4 and q_comp in by_artist_compact:
            return _pick_artist_rep_track(by_artist_compact[q_comp], prefer_tr=prefer_tr)

        # 5. Hybrid online search (Deezer + iTunes TR)
        online = search_online_tracks(clean_q, limit=5, prefer_tr=prefer_tr)
        if online:
            # Prefer online results matching q directly or matching TR scene
            def _on_rank(item):
                ia = _tr_fold(item.get("artist_name") or "")
                it = _tr_fold(item.get("title") or "")
                s = 0
                if ia == q_norm or it == q_norm:
                    s += 100
                if q_norm in ia or q_norm in it:
                    s += 40
                if item.get("region") == "tr":
                    s += 60 if prefer_tr else 20
                return s

            top = max(online, key=_on_rank)
            tid = top["id"]
            if tid in by_id:
                return by_id[tid]
            dkey = dedupe_key(top["artist_name"], top["title"])
            if dkey in by_key:
                return by_key[dkey]

            art_fold = _tr_fold(top["artist_name"] or "")
            artist_tracks = by_artist.get(art_fold, [])
            if artist_tracks:
                if top.get("region") == "tr" or prefer_tr:
                    tr_only = [(i, m) for i, m in artist_tracks if m.get("region") == "tr"]
                    if tr_only:
                        artist_tracks = tr_only
                top_words = set(_tr_fold(top["title"] or "").split())
                best_idx, best_m = None, None
                best_rank = (-1, -1)
                for idx, m in artist_tracks:
                    m_words = set(_TIT_FOLD[idx].split())
                    ov = len(top_words & m_words)
                    if ov > 0:
                        rk = (ov, 1 if (int(_NOISE_FLAGS[idx]) & 7) == 0 else 0)
                        if rk > best_rank:
                            best_rank = rk
                            best_idx, best_m = idx, m
                if best_idx is None:
                    best_idx, best_m = _pick_artist_rep_track(
                        artist_tracks, prefer_tr=prefer_tr, target_year=top.get("year")
                    )
                return best_idx, self._make_seed_proxy(best_m, top)

            # 5b. Out-of-catalog artist: resolve via Deezer related artist proxy instead of hard 404
            rel_proxy = _find_deezer_artist_proxy(top["artist_name"], top.get("artist_id"), prefer_tr=prefer_tr)
            if rel_proxy:
                return rel_proxy[0], self._make_seed_proxy(rel_proxy[1], top)

        # 6. Catalog exact title fallback
        title_hits = [(idx, meta_list[idx]) for idx in range(_N_TRACKS) if _TIT_FOLD[idx] == q_norm]
        if title_hits:
            return _pick_artist_rep_track(title_hits, prefer_tr=prefer_tr)

        return None, None

    def _send_json(self, data, code=200):
        try:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._send_cors_headers(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except Exception as e:
            print(f"Error sending json: {e}", flush=True)


if __name__ == "__main__":
    print(f"\n=======================================================")
    print(f"🚀 Suggestify Render Sunucusu Dinlemede: http://0.0.0.0:{PORT}")
    print(f"=======================================================\n")
    server = ThreadedHTTPServer(("0.0.0.0", PORT), RequestHandler)
    server.serve_forever()
