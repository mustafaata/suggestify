"""Ultra-lightweight Standalone Suggestify Web App & API for Render.com.
Zero-DB, Pure Numpy In-Memory Vector Engine (~145 MB RAM footprint).
"""
from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
import numpy as np
import requests

PORT = int(os.environ.get("PORT", 10000))

THEME_LABELS = {
    "heartbreak_regret": "Ayrılık, pişmanlık ve özlem",
    "love_romance": "Romantik aşk, sevgi ve tutku",
    "joy_carefree": "Neşeli, kutlama ve keyifli",
    "melancholy_lonely": "Yalnızlık, karanlık ve içsel hüzün",
    "anger_rebellion": "Öfke, isyan ve sert tavır",
    "street_flex": "Sokak, özgüven ve başarı",
    "neutral": "Dengeli / nötr anlatım",
}


@dataclass
class Params:
    alpha: float = 0.20  # Timbre (EffNet + CLAP)
    zeta: float = 0.25   # Arrangement signature
    beta: float = 0.26   # Audio DNA
    gamma: float = 0.08  # Genre
    eta: float = 0.10    # Lyrics theme
    k: int = 8
    strict_region: bool = False


MODES = {
    "emotional": Params(alpha=0.20, zeta=0.25, beta=0.26, gamma=0.08, eta=0.10, strict_region=False),
    "balanced": Params(alpha=0.18, zeta=0.22, beta=0.24, gamma=0.10, eta=0.10, strict_region=False),
    "local_scene": Params(alpha=0.18, zeta=0.22, beta=0.24, gamma=0.10, eta=0.10, strict_region=True),
}


def dedupe_key(artist: str, title: str) -> str:
    a = (artist or "").casefold().strip()
    t = (title or "").casefold().strip()
    return f"{a}::{t}"


print("⏳ [Render] Kompakt katalog yükleniyor (catalog_compact.npz)...")
t0 = time.time()
base_dir = os.path.dirname(__file__)
npz_path = os.path.join(base_dir, "catalog_compact.npz")
meta_path = os.path.join(base_dir, "catalog_meta.json.gz")

data = np.load(npz_path)
matrices = {k: data[k] for k in data.files}

with gzip.open(meta_path, "rt", encoding="utf-8") as f:
    meta_list = json.load(f)

print(f"✅ Katalog hazır: {len(meta_list)} parça ({time.time() - t0:.2f}s) | RAM: ~145 MB")

# Fast index
by_id = {int(m["id"]): (idx, m) for idx, m in enumerate(meta_list)}
by_key = {dedupe_key(m["artist_name"], m["title"]): (idx, m) for idx, m in enumerate(meta_list)}


def deezer_search(q: str):
    try:
        r = requests.get(f"https://api.deezer.com/search?q={requests.utils.quote(q)}&limit=3", timeout=3)
        return r.json().get("data", [])
    except Exception:
        return []


def deezer_track(tid: int):
    try:
        r = requests.get(f"https://api.deezer.com/track/{tid}", timeout=3)
        return r.json()
    except Exception:
        return {}


def _z(x, mask):
    v = x[mask]
    std = v.std()
    return (x - v.mean()) / (std + 1e-7) if std > 1e-7 else np.zeros_like(x)


def recommend_compact(seed_idx: int, params: Params):
    N = len(meta_list)
    mask = np.ones(N, dtype=bool)
    mask[seed_idx] = False

    seed_meta = meta_list[seed_idx]
    seed_artist = (seed_meta["artist_name"] or "").casefold()

    for i, m in enumerate(meta_list):
        if (m["artist_name"] or "").casefold() == seed_artist:
            mask[i] = False
        elif params.strict_region and seed_meta.get("region") != m.get("region"):
            mask[i] = False

    # 1. Texture & Timbre similarity (EffNet + CLAP)
    clap_sim = matrices["C"] @ matrices["C"][seed_idx]
    if matrices["has_effnet"][seed_idx]:
        eff_sim = matrices["E"] @ matrices["E"][seed_idx]
        tex_sim = np.where(matrices["has_effnet"], 0.56 * eff_sim + 0.44 * clap_sim, clap_sim)
    else:
        tex_sim = clap_sim

    # 2. Arrangement & Sonic Signature ('Şarkının İmzası')
    sz = matrices["S_z"][seed_idx]
    sig_sim = np.clip(1.0 - np.maximum(0.0, sz[None, :] - matrices["S_z"]) / 1.85, 0.0, 1.0).mean(axis=1)

    # 3. Audio DNA similarity
    D = np.abs(matrices["F"] - matrices["F"][seed_idx])
    feat_sim = 1.0 - (D @ np.ones(matrices["F"].shape[1])) / matrices["F"].shape[1]

    # 4. Genre & Style family similarity
    genre_sim = 0.54 * (matrices["G"] @ matrices["G"][seed_idx]) + 0.46 * (matrices["P"] @ matrices["P"][seed_idx])

    # 5. Lyrics Theme similarity
    lyr_sim = matrices["L_themes"] @ matrices["L_themes"][seed_idx]

    # Combined score
    scores = (
        params.alpha * _z(tex_sim, mask)
        + params.zeta * _z(sig_sim, mask)
        + params.beta * _z(feat_sim, mask)
        + params.gamma * _z(genre_sim, mask)
        + params.eta * _z(lyr_sim, mask)
    )
    scores[~mask] = -np.inf

    # Pick top-k
    order = np.argsort(-scores)[: params.k * 3]
    chosen = []
    seen_artists = set()
    for idx in order:
        if not mask[idx]:
            continue
        art = (meta_list[idx]["artist_name"] or "").casefold()
        if art in seen_artists:
            continue
        seen_artists.add(art)
        chosen.append((idx, float(scores[idx]), float(tex_sim[idx]), float(feat_sim[idx])))
        if len(chosen) >= params.k:
            break

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
.search-row { display: flex; gap: 0.75rem; }
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
  <div class="tagline">Müzikal Ruh ve Ses İkizleri — 22.457 Şarkılık Duygu Odaklı Öneri Motoru</div>
</header>

<main>
  <div class="search-box">
    <div class="search-row">
      <input type="text" id="queryInput" placeholder="Bir şarkı veya sanatçı yazın... (örn: Wegh, Adele, Altın Gün, Mor ve Ötesi)" autocomplete="off">
      <button class="btn-search" onclick="doSearch()">Keşfet</button>
    </div>
    <div class="modes-row">
      <span class="mode-label">Keşif Modu:</span>
      <button class="mode-btn active" data-mode="emotional" onclick="setMode('emotional', this)">🌟 Duygu & His (Sınır Tanımayan)</button>
      <button class="mode-btn" data-mode="balanced" onclick="setMode('balanced', this)">⚖️ Dengeli Hibrit</button>
      <button class="mode-btn" data-mode="local_scene" onclick="setMode('local_scene', this)">🇹🇷 Sadece Yerel Sahne</button>
    </div>
    <div class="quick-picks">
      <span class="mode-label">Hızlı Örnekler:</span>
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
let currentPlayBtn = null;

function setMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const q = document.getElementById('queryInput').value.trim();
  if (q) doSearch();
}

function quick(q) {
  document.getElementById('queryInput').value = q;
  doSearch();
}

document.getElementById('queryInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') doSearch();
});

async function doSearch() {
  const q = document.getElementById('queryInput').value.trim();
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

function renderResults(data) {
  const seed = data.seed;
  const recs = data.recommendations;

  const seedCover = seed.cover_url || 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=200&auto=format&fit=crop&q=80';
  document.getElementById('seedContainer').innerHTML = `
    <div class="seed-card">
      <img src="${seedCover}" class="seed-cover" alt="cover">
      <div class="seed-info">
        <div class="seed-title">${seed.title}</div>
        <div class="seed-artist">${seed.artist_name} · ${seed.year || ''} (${seed.region ? seed.region.toUpperCase() : 'TR'})</div>
        <div class="pill-group">
          <span class="pill">Tempo: ${seed.bpm} BPM</span>
          <span class="pill">${seed.key}</span>
          ${seed.lyric_dominant ? `<span class="pill lyric">Söz: ${seed.lyric_label}</span>` : ''}
          ${seed.signatures.map(s => `<span class="pill sig">${s}</span>`).join('')}
        </div>
      </div>
      ${seed.preview ? `<button class="btn-play" onclick="playAudio('${seed.preview}', '${seed.title.replace(/'/g, "\\'")}', '${seed.artist_name.replace(/'/g, "\\'")}', this)">▶</button>` : ''}
    </div>
  `;

  document.getElementById('recsCount').innerText = `${recs.length} Şarkı Eşleşti`;
  const list = document.getElementById('recList');
  list.innerHTML = recs.map((r, i) => {
    const cover = r.cover_url || 'https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=120&auto=format&fit=crop&q=80';
    const regClass = r.region === 'tr' ? 'rec-region-tr' : 'rec-region-global';
    return `
      <div class="rec-card">
        <div class="rec-rank">#${i+1}</div>
        <img src="${cover}" class="rec-cover" alt="cover">
        <div class="rec-main">
          <div class="rec-title-row">
            <span class="rec-title">${r.title}</span>
            <span class="rec-year">${r.year || ''}</span>
            <span class="rec-region-tag ${regClass}">${r.region}</span>
          </div>
          <div class="rec-artist">${r.artist_name}</div>
          <div class="rec-why">
            ${r.because.map(b => `<span class="why-tag">${b}</span>`).join('')}
          </div>
        </div>
        <div class="rec-scores">
          <div class="dna-match">%${Math.round(r.dna * 100)}</div>
          <div class="dna-sub">Ses Uyumu</div>
        </div>
        ${r.preview ? `<button class="btn-play" onclick="playAudio('${r.preview}', '${r.title.replace(/'/g, "\\'")}', '${r.artist_name.replace(/'/g, "\\'")}', this)">▶</button>` : ''}
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


class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/recommend":
            self.handle_recommend(parse_qs(parsed.query))
        elif parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/health":
            self._send_json({"status": "healthy", "tracks": len(meta_list)})
        else:
            self.send_response(404)
            self.end_headers()

    def handle_recommend(self, params):
        q = params.get("query", [""])[0]
        mode_name = params.get("mode", ["emotional"])[0]
        k = int(params.get("k", [8])[0])

        if not q:
            self._send_json({"error": "Sorgu belirtilmedi"}, 400)
            return

        try:
            seed_idx, seed_meta = self._resolve(q)
            if seed_idx is None:
                self._send_json({"error": f"'{q}' bulunamadı"}, 404)
                return

            params_obj = replace(MODES.get(mode_name, MODES["emotional"]), k=k)
            recs = recommend_compact(seed_idx, params_obj)

            dom_theme = seed_meta.get("lyric_dominant") or "neutral"
            preview_url = seed_meta.get("preview")
            if not preview_url:
                dt = deezer_track(int(seed_meta["id"]))
                preview_url = dt.get("preview")

            seed_data = {
                "id": seed_meta["id"],
                "title": seed_meta["title"],
                "artist_name": seed_meta["artist_name"],
                "year": seed_meta.get("year"),
                "region": seed_meta.get("region"),
                "cover_url": seed_meta.get("cover_url"),
                "preview": preview_url,
                "bpm": round(float(seed_meta.get("bpm") or 120)),
                "key": f"{seed_meta.get('key') or 'C'} {seed_meta.get('scale') or 'major'}",
                "lyric_dominant": dom_theme,
                "lyric_label": THEME_LABELS.get(dom_theme, dom_theme),
                "signatures": seed_meta.get("sig_traits", []),
            }

            rec_list = []
            for idx, score, tex_sim, feat_sim in recs:
                m = meta_list[idx]
                p_url = m.get("preview")
                if not p_url:
                    dt = deezer_track(int(m["id"]))
                    p_url = dt.get("preview")

                reasons = m.get("sig_traits", [])[:2]
                rec_list.append({
                    "id": m["id"],
                    "artist_name": m["artist_name"],
                    "title": m["title"],
                    "year": m.get("year"),
                    "region": m.get("region"),
                    "cover_url": m.get("cover_url"),
                    "preview": p_url,
                    "because": reasons or ["tınısal uyum"],
                    "score": round(score, 2),
                    "dna": round(feat_sim, 2),
                })

            self._send_json({"seed": seed_data, "recommendations": rec_list})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _resolve(self, q):
        q_norm = q.strip().casefold()
        for key, val in by_key.items():
            if q_norm in key or key in q_norm:
                return val
        for idx, m in enumerate(meta_list):
            if q_norm in (m["artist_name"] or "").casefold():
                return idx, m
        results = deezer_search(q)
        if results:
            t = results[0]
            tid = int(t["id"])
            if tid in by_id:
                return by_id[tid]
            dkey = dedupe_key(t["artist"]["name"], t.get("title_short") or t["title"])
            if dkey in by_key:
                return by_key[dkey]
        return 0, meta_list[0]

    def _send_json(self, data, code=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    print(f"\n=======================================================")
    print(f"🚀 Suggestify Render Sunucusu Dinlemede: http://0.0.0.0:{PORT}")
    print(f"=======================================================\n")
    server = ThreadedHTTPServer(("0.0.0.0", PORT), RequestHandler)
    server.serve_forever()
