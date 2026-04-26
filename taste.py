"""
Taste — Emotion Fingerprint Music Recommender
==============================================
Find songs that make you feel exactly the same way.

Requirements:
    pip install streamlit librosa numpy scipy scikit-learn spotipy soundfile pydub python-dotenv pandas joblib matplotlib requests

Run with:
    streamlit run taste.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import os
import json
import sqlite3
import uuid
import io
import requests
import tempfile
from typing import Optional  # FIX #4: replaced `bytes | None` with Optional[bytes]

import librosa
import librosa.effects
import librosa.feature
import librosa.onset
import librosa.beat

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from scipy.spatial.distance import cosine
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def get_config(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, fallback)

SPOTIFY_CLIENT_ID     = get_config("SPOTIFY_CLIENT_ID",     "YOUR_CLIENT_ID_HERE")
SPOTIFY_CLIENT_SECRET = get_config("SPOTIFY_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")
SPOTIFY_REDIRECT_URI  = get_config("SPOTIFY_REDIRECT_URI",  "http://localhost:8501")

DB_PATH              = "taste.db"
MODEL_PATH           = "taste_model.pkl"
MIN_SAMPLES_TO_TRAIN = 5

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            track_id    TEXT,
            track_name  TEXT,
            artist      TEXT,
            rating      INTEGER,
            feature_vec TEXT,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(session_id, track_id, track_name, artist, rating, feature_vec):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback (session_id, track_id, track_name, artist, rating, feature_vec)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, track_id, track_name, artist, rating,
          json.dumps(feature_vec.tolist())))
    conn.commit()
    conn.close()


def load_all_feedback():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT feature_vec, rating FROM feedback WHERE rating IS NOT NULL")
    rows = c.fetchall()
    conn.close()
    return [(np.array(json.loads(fv), dtype=np.float32), r) for fv, r in rows]


def get_feedback_count():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM feedback WHERE rating IS NOT NULL")
    count = c.fetchone()[0]
    conn.close()
    return count


# ══════════════════════════════════════════════════════════════════════════════
# SPOTIFY CLIENT
# ══════════════════════════════════════════════════════════════════════════════

# FIX #5: Cache the Spotify client so it isn't re-created on every rerun
@st.cache_resource
def get_spotify_client():
    auth_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def search_spotify_songs(sp, query: str, limit: int = 10) -> list:
    """Search Spotify for songs by query string."""
    if not query:
        return []
    try:
        results = sp.search(q=query, type='track', limit=limit)
        tracks = results.get('tracks', {}).get('items', [])
        return [{
            'id':           t['id'],
            'name':         t['name'],
            'artist':       ', '.join(a['name'] for a in t['artists']),
            'preview_url':  t.get('preview_url'),
            'external_url': t['external_urls'].get('spotify', ''),
            'album_art':    t['album']['images'][0]['url'] if t['album']['images'] else None,
            'duration_ms':  t['duration_ms'],
            'popularity':   t.get('popularity', 0),
        } for t in tracks]
    except Exception as e:
        st.error(f"Spotify search error: {e}")
        return []


def download_preview(preview_url: str) -> Optional[bytes]:  # FIX #4
    """Download a Spotify 30s preview MP3 and return as bytes."""
    try:
        response = requests.get(preview_url, timeout=10)
        if response.status_code == 200:
            return response.content
        return None
    except Exception:
        return None


def get_soundcloud_audio(query: str) -> Optional[bytes]:  # FIX #4
    """Search SoundCloud and download the audio for a given song query."""
    try:
        import yt_dlp
        search_query = f"scsearch1:{query}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': '/tmp/taste_audio.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([search_query])

        with open('/tmp/taste_audio.mp3', 'rb') as f:
            return f.read()
    except Exception as e:
        st.error(f"SoundCloud download failed: {e}")
        return None


def search_tracks_by_features(sp, features: dict, n_results: int = 50) -> list:
    """Search Spotify for recommendations based on audio features."""
    tempo      = features.get('tempo', 120.0)
    rms_mean   = features.get('rms_mean', 0.05)
    harm_ratio = features.get('harmonic_ratio', 0.5)

    if tempo > 120:
        energy = "energetic upbeat"
    elif tempo > 90:
        energy = "moderate tempo"
    else:
        energy = "slow calm"

    if harm_ratio > 0.6:
        mood = "melodic harmonic"
    else:
        mood = "rhythmic"

    queries = [
        f"{energy} {mood}",
        f"{energy} music",
        f"{mood} music",
        "popular music",
        "top hits"
    ]

    results  = []
    seen_ids = set()

    for query in queries:
        if len(results) >= n_results:
            break
        try:
            search_results = sp.search(q=query, type='track', limit=10)
            tracks = search_results.get('tracks', {}).get('items', [])
            for t in tracks:
                if t['id'] not in seen_ids:
                    seen_ids.add(t['id'])
                    results.append({
                        'id':           t['id'],
                        'name':         t['name'],
                        'artist':       ', '.join(a['name'] for a in t['artists']),
                        'preview_url':  t.get('preview_url'),
                        'external_url': t['external_urls'].get('spotify', ''),
                        'album_art':    t['album']['images'][0]['url'] if t['album']['images'] else None,
                        'popularity':   t.get('popularity', 0),
                        '_target_tempo':        tempo,
                        '_target_energy':       float(np.clip(rms_mean * 10, 0, 1)),
                        '_target_valence':      float(np.clip(harm_ratio, 0, 1)),
                        '_target_dance':        float(np.clip(tempo / 200, 0, 1)),
                        '_target_acoustic':     float(np.clip(1.0 - (features.get('spectral_centroid_mean', 2000) / 8000), 0, 1)),
                        '_target_instrumental': float(np.clip(1.0 - features.get('zcr_mean', 0.05) * 10, 0, 1)),
                    })
        except Exception as e:
            # FIX #11: surface errors instead of silently swallowing them
            st.warning(f"Search query '{query}' failed: {e}")
            continue

    return results[:n_results]


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def load_audio_segment(file_bytes: bytes, start_sec: float,
                       end_sec: float, sr: int = 22050) -> tuple:
    buffer = io.BytesIO(file_bytes)
    y, sr = librosa.load(buffer, sr=sr, offset=start_sec,
                         duration=(end_sec - start_sec), mono=True)
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> dict:
    features = {}

    # Amplitude
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std']  = float(np.std(rms))
    features['rms_max']  = float(np.max(rms))

    # Rhythm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flatten()[0])
    features['tempo'] = tempo

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(beats) > 0:
        valid_beats = beats[beats < len(onset_env)]
        beat_strengths = onset_env[valid_beats]
        features['beat_strength_mean'] = float(np.mean(beat_strengths))
        features['beat_strength_std']  = float(np.std(beat_strengths))
    else:
        features['beat_strength_mean'] = 0.0
        features['beat_strength_std']  = 0.0

    if len(beats) > 1:
        ibi = np.diff(librosa.frames_to_time(beats, sr=sr))
        features['beat_regularity'] = float(1.0 / (np.std(ibi) + 1e-6))
    else:
        features['beat_regularity'] = 0.0

    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_means = np.mean(chroma, axis=1)
    for i, val in enumerate(chroma_means):
        features[f'chroma_{i}'] = float(val)
    features['chroma_energy'] = float(np.sum(chroma_means))
    features['chroma_std']    = float(np.std(chroma_means))

    # Tonnetz
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    for i, val in enumerate(np.mean(tonnetz, axis=1)):
        features[f'tonnetz_{i}'] = float(val)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i}_std']  = float(np.std(mfcc[i]))

    # Spectral
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spec_centroid))
    features['spectral_centroid_std']  = float(np.std(spec_centroid))

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean']  = float(np.mean(spec_rolloff))

    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(spec_bandwidth))

    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(spec_contrast.shape[0]):
        features[f'spectral_contrast_{i}'] = float(np.mean(spec_contrast[i]))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std']  = float(np.std(zcr))

    # Harmonic/Percussive
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy  = float(np.mean(y_harm ** 2))
    perc_energy  = float(np.mean(y_perc ** 2))
    total_energy = harm_energy + perc_energy + 1e-10
    features['harmonic_ratio']   = harm_energy / total_energy
    features['percussive_ratio'] = perc_energy / total_energy

    return features


# FIX #7: Sort keys so vector ordering is always deterministic
def features_to_vector(features: dict) -> np.ndarray:
    return np.array([features[k] for k in sorted(features.keys())], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ML MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=42
        ))
    ])


def train_model(feedback_data):
    if len(feedback_data) < MIN_SAMPLES_TO_TRAIN:
        return None
    X = np.nan_to_num(np.array([fv for fv, _ in feedback_data]), nan=0.0)
    y = np.array([r for _, r in feedback_data], dtype=float)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def predict_score(pipeline, feature_vec):
    if pipeline is None:
        return 5.0
    fv = np.nan_to_num(feature_vec, nan=0.0).reshape(1, -1)
    return float(np.clip(pipeline.predict(fv)[0], 1.0, 10.0))


def get_feature_weights(pipeline, n_features: int) -> np.ndarray:
    if pipeline is None:
        return np.ones(n_features, dtype=np.float32)
    importances = pipeline.named_steps['model'].feature_importances_
    # FIX #8: safely handle length mismatch between importances and n_features
    weights = np.ones(n_features, dtype=np.float32)
    min_len = min(len(importances), n_features)
    weights[:min_len] = (importances[:min_len] / (importances[:min_len].mean() + 1e-10))
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def weighted_cosine_similarity(vec_a, vec_b, weights):
    wa, wb = vec_a * weights, vec_b * weights
    if np.linalg.norm(wa) == 0 or np.linalg.norm(wb) == 0:
        return 0.0
    return float(1.0 - cosine(wa, wb))


# FIX #7: Use sorted keys to match features_to_vector ordering
def spotify_track_to_vector(track: dict, ref_features: dict) -> np.ndarray:
    sorted_keys  = sorted(ref_features.keys())
    n            = len(sorted_keys)
    vec          = np.zeros(n, dtype=np.float32)
    key_to_index = {k: i for i, k in enumerate(sorted_keys)}

    def set_feat(name, value):
        if name in key_to_index:
            vec[key_to_index[name]] = float(value)

    set_feat('tempo',                  track.get('_target_tempo', 120.0))
    set_feat('rms_mean',               track.get('_target_energy', 0.5) * 0.1)
    set_feat('harmonic_ratio',         track.get('_target_valence', 0.5) * 0.8)
    set_feat('beat_regularity',        track.get('_target_dance', 0.5) * 50)
    set_feat('spectral_centroid_mean', (1.0 - track.get('_target_acoustic', 0.5)) * 8000)
    set_feat('zcr_mean',               (1.0 - track.get('_target_instrumental', 0.5)) * 0.1)
    return vec


def rank_candidates(candidates, ref_feature_dict, ref_feature_vec, ml_pipeline, weights):
    scored = []
    for track in candidates:
        cand_vec = spotify_track_to_vector(track, ref_feature_dict)
        sim      = weighted_cosine_similarity(ref_feature_vec, cand_vec, weights)
        ml_score = predict_score(ml_pipeline, cand_vec)
        combined = 0.6 * sim + 0.4 * (ml_score / 10.0)
        t = track.copy()
        t['similarity_score']   = round(sim * 100, 1)
        t['ml_predicted_score'] = round(ml_score, 1)
        t['combined_score']     = round(combined * 100, 1)
        scored.append(t)
    scored.sort(key=lambda x: x['combined_score'], reverse=True)
    return scored


def generate_session_id():
    return str(uuid.uuid4())[:8]


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Taste 🎵",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main { background-color: #0a0a0a; }
        .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #111827 100%); }
        .taste-header { text-align: center; padding: 2rem 0 1rem 0; }
        .taste-title {
            font-size: 4rem; font-weight: 700;
            background: linear-gradient(90deg, #1db954, #1ed760, #17a844);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -2px;
        }
        .taste-subtitle { font-size: 1.1rem; color: #9ca3af; margin-top: -0.5rem; }
        .step-header { font-size: 1.4rem; font-weight: 600; color: #ffffff; margin: 1.5rem 0 0.5rem 0; }
        .stButton > button {
            background: linear-gradient(90deg, #1db954, #17a844);
            color: white; border: none; border-radius: 50px;
            font-weight: 600; padding: 0.5rem 2rem; transition: all 0.2s ease;
        }
        .stButton > button:hover { transform: scale(1.02); box-shadow: 0 4px 20px rgba(29,185,84,0.4); }
        div[data-testid="metric-container"] {
            background: #1a1a2e; border: 1px solid #2d2d2d; border-radius: 12px; padding: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Init ──────────────────────────────────────────────────────────────────
    init_db()

    defaults = {
        'session_id':      generate_session_id(),
        'recommendations': [],
        'ref_features':    None,
        'ref_vec':         None,
        'feedback_given':  {},
        'ml_pipeline':     load_model(),
        'selected_track':  None,
        'preview_bytes':   None,
        'search_results':  [],
        'search_query':    '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    sp = get_spotify_client()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎵 Taste")
        st.caption("Songs that feel the same way.")
        st.divider()

        st.subheader("🤖 ML Status")
        count = get_feedback_count()
        st.metric("Ratings collected", count)
        st.progress(min(count / MIN_SAMPLES_TO_TRAIN, 1.0))

        if count >= MIN_SAMPLES_TO_TRAIN:
            st.success("✅ Personalised to you!")
        else:
            st.info(f"Rate {MIN_SAMPLES_TO_TRAIN - count} more to activate ML")

        if count >= MIN_SAMPLES_TO_TRAIN:
            if st.button("🔄 Retrain Model", use_container_width=True):
                pipeline = train_model(load_all_feedback())
                if pipeline:
                    st.session_state.ml_pipeline = pipeline
                    st.success("Retrained!")

        st.divider()
        st.markdown("""
        **How it works:**
        1. 🔍 Search for a song
        2. ▶️ Listen to the preview
        3. ✂️ Pick the part that hits
        4. 🎯 Get matched songs
        5. ⭐ Rate to improve results
        """)
        st.divider()
        st.caption(f"Session: `{st.session_state.session_id}`")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="taste-header">
            <div class="taste-title">taste</div>
            <div class="taste-subtitle">find songs that make you feel exactly the same way</div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — SEARCH FOR A SONG
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-header">🔍 Step 1 — Search for a Song</div>',
                unsafe_allow_html=True)

    col_search, col_btn = st.columns([5, 1])
    with col_search:
        query = st.text_input(
            "Search",
            placeholder="Search for any song or artist...",
            label_visibility="collapsed"
        )
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked and query:
        with st.spinner("Searching Spotify..."):
            st.session_state.search_results = search_spotify_songs(sp, query, limit=8)
            st.session_state.search_query   = query

   # ── Search Results ────────────────────────────────────────────────────────────
if st.session_state.search_results:
    st.markdown("**Select a song:**")

    for i, track in enumerate(st.session_state.search_results):
        col_art, col_info, col_select = st.columns([1, 6, 2])

        with col_art:
            if track.get('album_art'):
                st.image(track['album_art'], width=55)

        with col_info:
            st.markdown(f"**{track['name']}**")
            st.caption(track['artist'])

        with col_select:
            # ✅ One button, index-based key, guaranteed unique
            if st.button("Select ✓", key=f"select_{i}", use_container_width=True):

                # ✅ Clear search results FIRST so loop won't
                # re-render these buttons on the next rerun
                st.session_state.search_results  = []
                st.session_state.selected_track  = track
                st.session_state.recommendations = []
                st.session_state.preview_bytes   = None  # reset stale audio

                if track.get('preview_url'):
                    with st.spinner("Loading preview..."):
                        preview = download_preview(track['preview_url'])
                        st.session_state.preview_bytes = preview
                        if not preview:
                            st.warning("⚠️ Preview download failed, trying SoundCloud...")
                            sc_query = f"{track['name']} {track['artist']}"
                            audio = get_soundcloud_audio(sc_query)
                            st.session_state.preview_bytes = audio
                else:
                    # No Spotify preview — go straight to SoundCloud
                    sc_query = f"{track['name']} {track['artist']}"
                    with st.spinner("🔍 Finding audio on SoundCloud..."):
                        audio = get_soundcloud_audio(sc_query)
                        if audio:
                            st.session_state.preview_bytes = audio
                        else:
                            st.warning("⚠️ No audio found. Try uploading the file manually.")

                st.rerun()

    # ── File upload fallback ──────────────────────────────────────────────────
    with st.expander("📁 Or upload your own file instead"):
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=["mp3", "wav", "flac", "ogg", "m4a", "mpeg", "mp4"],
        )
        if uploaded_file:
            # FIX #10: set preview_bytes before selected_track to avoid flash
            file_bytes = uploaded_file.read()
            st.session_state.preview_bytes  = file_bytes
            st.session_state.selected_track = {
                'id':           'uploaded',
                'name':         uploaded_file.name,
                'artist':       'Uploaded file',
                'preview_url':  None,
                'external_url': '',
                'album_art':    None,
                'duration_ms':  0,
            }
            st.success(f"✅ Loaded: {uploaded_file.name}")
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — SELECTED SONG + PICK YOUR PART
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.selected_track:
        track = st.session_state.selected_track
        st.divider()

        col_art, col_info = st.columns([1, 6])
        with col_art:
            if track.get('album_art'):
                st.image(track['album_art'], width=80)
        with col_info:
            st.markdown(f"### {track['name']}")
            st.caption(f"**{track['artist']}**")
            if track.get('external_url'):
                st.markdown(f"[Open on Spotify ↗]({track['external_url']})")

        if st.session_state.preview_bytes:
            st.audio(st.session_state.preview_bytes, format="audio/mpeg")
            st.caption("⬆️ Listen to the preview above to find your favourite part")
        else:
            st.warning("⚠️ No preview available — try another version or upload the file directly")

        st.divider()

        st.markdown('<div class="step-header">✂️ Step 2 — Pick the Part That Hits</div>',
                    unsafe_allow_html=True)
        st.markdown("Set the **start and end time** of the 10–20 second moment you want to match.")

        # FIX #6: correct max_duration logic
        if track.get('preview_url') and track['id'] != 'uploaded':
            max_duration = 30.0
            st.info("💡 Spotify previews are 30 seconds — pick your segment within that window")
        elif track.get('duration_ms', 0) > 0:
            max_duration = track['duration_ms'] / 1000
        else:
            max_duration = 300.0

        col1, col2 = st.columns(2)
        with col1:
            start_sec = st.number_input(
                "⏱ Start time (seconds)",
                min_value=0.0,
                max_value=float(max(max_duration - 10, 0.0)),
                value=0.0,
                step=0.5
            )
        with col2:
            end_sec = st.number_input(
                "⏱ End time (seconds)",
                min_value=float(start_sec + 10.0),
                max_value=float(min(start_sec + 20.0, max_duration)),
                # FIX #6: safe default that never exceeds max_duration
                value=float(min(start_sec + 15.0, max_duration)),
                step=0.5
            )

        duration = end_sec - start_sec

        # FIX #9: validate BEFORE entering spinner so st.stop() is safe
        if duration < 10.0:
            st.warning(f"⚠️ Too short ({duration:.1f}s) — minimum is 10 seconds")
            st.stop()
        elif duration > 20.0:
            st.warning(f"⚠️ Too long ({duration:.1f}s) — maximum is 20 seconds")
            st.stop()
        else:
            st.success(f"✅ Selected: **{start_sec:.1f}s → {end_sec:.1f}s** ({duration:.1f}s)")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3 — ANALYSE
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="step-header">🔬 Step 3 — Find Your Match</div>',
                    unsafe_allow_html=True)

        if st.button("🚀 Analyse & Find Matches", type="primary",
                     use_container_width=True):

            # FIX #9: guard checks BEFORE spinners so st.stop() never fires inside one
            if not st.session_state.preview_bytes:
                st.error("❌ No audio to analyse — please select a song with a preview "
                         "or upload a file")
                st.stop()

            features  = None
            feat_vec  = None
            candidates = []

            with st.spinner("🎼 Analysing your audio fingerprint..."):
                try:
                    y, sr    = load_audio_segment(
                        st.session_state.preview_bytes, start_sec, end_sec
                    )
                    features = extract_features(y, sr)
                    feat_vec = features_to_vector(features)
                    st.session_state.ref_features = features
                    st.session_state.ref_vec      = feat_vec
                except Exception as e:
                    st.error(f"❌ Audio analysis failed: {e}")

            # FIX #9: stop after spinner exits, not inside it
            if features is None:
                st.stop()

            # Feature summary
            with st.expander("🔬 Your Audio Fingerprint", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🥁 Tempo",       f"{features['tempo']:.1f} BPM")
                c2.metric("⚡ Energy",       f"{features['rms_mean']:.4f}")
                c3.metric("🎸 Harmonic",     f"{features['harmonic_ratio']:.1%}")
                c4.metric("🥁 Percussive",   f"{features['percussive_ratio']:.1%}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("💓 Beat Strength",   f"{features['beat_strength_mean']:.3f}")
                c6.metric("🌈 Spectral Centre", f"{features['spectral_centroid_mean']:.0f} Hz")
                c7.metric("🎯 Beat Regularity", f"{features['beat_regularity']:.1f}")
                c8.metric("〰️ Texture",         f"{features['zcr_mean']:.4f}")

                st.markdown("**MFCC Timbre Fingerprint:**")
                st.bar_chart({f"MFCC {i}": features[f'mfcc_{i}_mean'] for i in range(13)})

                st.markdown("**Chroma (Pitch Classes):**")
                note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                st.bar_chart({note_names[i]: features[f'chroma_{i}'] for i in range(12)})

            with st.spinner("🎵 Searching for emotional matches..."):
                try:
                    candidates = search_tracks_by_features(sp, features, n_results=50)
                except Exception as e:
                    st.error(f"❌ Search error: {e}")

            if not candidates:
                st.warning("No matches found — check your Spotify credentials")
                st.stop()

            pipeline = st.session_state.ml_pipeline
            weights  = get_feature_weights(pipeline, len(feat_vec))
            ranked   = rank_candidates(candidates, features, feat_vec, pipeline, weights)

            st.session_state.recommendations = ranked[:10]
            st.session_state.feedback_given  = {}
            st.success("✅ Found your top **10** emotional matches!")

        # ══════════════════════════════════════════════════════════════════════
        # STEP 4 — RESULTS
        # ══════════════════════════════════════════════════════════════════════
        if st.session_state.recommendations:
            st.divider()
            st.markdown('<div class="step-header">🎧 Step 4 — Your Matches</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "Listen and rate how well each song makes you feel the **same way**. "
                "Your ratings teach Taste your personal emotional fingerprint."
            )

            for i, track in enumerate(st.session_state.recommendations):
                already_rated = track['id'] in st.session_state.feedback_given

                with st.container():
                    col_img, col_main, col_rate = st.columns([1, 5, 3])

                    with col_img:
                        if track.get('album_art'):
                            st.image(track['album_art'], width=80)
                        else:
                            st.markdown("🎵")

                    with col_main:
                        rank_emoji = ["🥇","🥈","🥉","4️⃣","5️⃣",
                                      "6️⃣","7️⃣","8️⃣","9️⃣","🔟"][i]
                        st.markdown(
                            f"{rank_emoji} **[{track['name']}]({track['external_url']})**  "
                            f"— *{track['artist']}*"
                        )
                        st.caption(
                            f"🎯 Similarity: **{track['similarity_score']}%** | "
                            f"🤖 ML: **{track['ml_predicted_score']}/10** | "
                            f"⭐ Score: **{track['combined_score']}%**"
                        )
                        if track.get('preview_url'):
                            st.audio(track['preview_url'], format='audio/mp3')
                        else:
                            st.caption("*No preview — open on Spotify to listen*")

                    with col_rate:
                        if already_rated:
                            r   = st.session_state.feedback_given[track['id']]
                            bar = "█" * r + "░" * (10 - r)
                            st.success(f"Rated: {r}/10\n`{bar}`")
                        else:
                            rating = st.slider(
                                "Same feeling?",
                                min_value=1, max_value=10, value=5,
                                key=f"slider_{track['id']}",
                                help="1 = totally different | 10 = identical feeling"
                            )
                            if st.button("✅ Rate",
                                         key=f"btn_{track['id']}",
                                         use_container_width=True):
                                save_feedback(
                                    st.session_state.session_id,
                                    track['id'], track['name'],
                                    track['artist'], rating,
                                    st.session_state.ref_vec
                                )
                                st.session_state.feedback_given[track['id']] = rating

                                total = get_feedback_count()
                                # FIX #12: guard against total==0 and only retrain every 5 NEW ratings
                                if total >= MIN_SAMPLES_TO_TRAIN and total > 0 and total % 5 == 0:
                                    new_pipeline = train_model(load_all_feedback())
                                    if new_pipeline:
                                        st.session_state.ml_pipeline = new_pipeline
                                        st.toast("🤖 ML model updated!")
                                st.rerun()

                    st.divider()

            # Session summary
            if st.session_state.feedback_given:
                st.subheader("📊 Your Ratings")
                rated = [(t, st.session_state.feedback_given[t['id']])
                         for t in st.session_state.recommendations
                         if t['id'] in st.session_state.feedback_given]
                rated.sort(key=lambda x: x[1], reverse=True)

                for track, r in rated:
                    bar = "█" * r + "░" * (10 - r)
                    ca, cb = st.columns([4, 1])
                    ca.markdown(f"**{track['name']}** — *{track['artist']}*  \n`{bar}`")
                    cb.metric("", f"{r}/10")

                avg = np.mean([r for _, r in rated])
                st.info(f"📈 Average feel-match: **{avg:.1f}/10** across {len(rated)} tracks")


if __name__ == "__main__":
    main()