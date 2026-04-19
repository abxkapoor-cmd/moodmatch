"""
MoodMatch — Single File Version
================================
Emotion-Fingerprint Music Recommender

Requirements (pip install these first):
    pip install streamlit librosa numpy scipy scikit-learn spotipy soundfile pydub python-dotenv pandas joblib matplotlib

Run with:
    streamlit run moodmatch.py

Setup:
    1. Get Spotify credentials from https://developer.spotify.com/dashboard
    2. Create a.env file with:
        SPOTIFY_CLIENT_ID=your_client_id
        SPOTIFY_CLIENT_SECRET=your_client_secret
        SPOTIFY_REDIRECT_URI=http://localhost:8501
    OR set them directly in the CONFIG section below.
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
import hashlib

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
# CONFIG — Set your Spotify credentials here if not using a.env file
# ══════════════════════════════════════════════════════════════════════════════

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "YOUR_CLIENT_ID_HERE")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501")

DB_PATH               = "moodmatch.db"
MODEL_PATH            = "moodmatch_model.pkl"
MIN_SAMPLES_TO_TRAIN  = 5

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER
# ══════════════════════════════════════════════════════════════════════════════

def init_db():
    """Initialise SQLite database with required tables."""
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

    c.execute("""
        CREATE TABLE IF NOT EXISTS clip_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_hash   TEXT UNIQUE,
            features    TEXT,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_feedback(session_id: str, track_id: str, track_name: str,
                  artist: str, rating: int, feature_vec: np.ndarray):
    """Save a user rating to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback (session_id, track_id, track_name, artist,
                              rating, feature_vec)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, track_id, track_name, artist, rating,
          json.dumps(feature_vec.tolist())))
    conn.commit()
    conn.close()


def load_all_feedback() -> list:
    """Load all feedback entries as (feature_vec, rating) tuples."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT feature_vec, rating FROM feedback WHERE rating IS NOT NULL")
    rows = c.fetchall()
    conn.close()
    result = []
    for fv_json, rating in rows:
        fv = np.array(json.loads(fv_json), dtype=np.float32)
        result.append((fv, rating))
    return result


def get_feedback_count() -> int:
    """Return total number of feedback ratings stored."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM feedback WHERE rating IS NOT NULL")
    count = c.fetchone()[0]
    conn.close()
    return count


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYSIS ENGINE (librosa)
# ══════════════════════════════════════════════════════════════════════════════

def load_audio_segment(file_bytes: bytes, start_sec: float,
                       end_sec: float, sr: int = 22050) -> tuple:
    """
    Load a specific time segment from raw audio bytes.
    Returns (waveform_array, sample_rate).
    """
    buffer = io.BytesIO(file_bytes)
    y, sr = librosa.load(
        buffer,
        sr=sr,
        offset=start_sec,
        duration=(end_sec - start_sec),
        mono=True
    )
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Extract a comprehensive ~70-dimensional audio feature set from a waveform.

    Features extracted:
      - Amplitude / Dynamics   : RMS mean, std, max
      - Rhythm                 : Tempo, beat strength, beat regularity
      - Pitch / Harmony        : Chroma (12 pitch classes), Tonnetz (6 dims)
      - Timbre                 : MFCCs (13 coefficients × mean + std)
      - Spectral               : Centroid, rolloff, bandwidth, contrast (7 bands)
      - Texture                : Zero-crossing rate
      - Structure              : Harmonic/percussive energy ratio
    """
    features = {}

    # ── 1. AMPLITUDE / DYNAMICS ──────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std']  = float(np.std(rms))
    features['rms_max']  = float(np.max(rms))

    # ── 2. RHYTHM / TEMPO ────────────────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)

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

    # ── 3. PITCH / HARMONY (Chroma + Tonnetz) ────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_means = np.mean(chroma, axis=1)
    for i, val in enumerate(chroma_means):
        features[f'chroma_{i}'] = float(val)
    features['chroma_energy'] = float(np.sum(chroma_means))
    features['chroma_std']    = float(np.std(chroma_means))

    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    for i, val in enumerate(np.mean(tonnetz, axis=1)):
        features[f'tonnetz_{i}'] = float(val)

    # ── 4. TIMBRE (MFCCs × 13) ───────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i}_std']  = float(np.std(mfcc[i]))

    # ── 5. SPECTRAL FEATURES ─────────────────────────────────────────────────
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

    # ── 6. ZERO CROSSING RATE ────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std']  = float(np.std(zcr))

    # ── 7. HARMONIC / PERCUSSIVE RATIO ───────────────────────────────────────
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy  = float(np.mean(y_harm ** 2))
    perc_energy  = float(np.mean(y_perc ** 2))
    total_energy = harm_energy + perc_energy + 1e-10
    features['harmonic_ratio']   = harm_energy / total_energy
    features['percussive_ratio'] = perc_energy / total_energy

    return features


def features_to_vector(features: dict) -> np.ndarray:
    """Convert feature dictionary to a flat numpy array."""
    return np.array(list(features.values()), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SPOTIFY CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def get_spotify_client() -> spotipy.Spotify:
    """Initialise and return an authenticated Spotify client."""
    auth_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def search_tracks_by_features(sp: spotipy.Spotify,
                               features: dict,
                               n_results: int = 50) -> list:
    """
    Map librosa audio features to Spotify tuneable attributes and
    call the Recommendations endpoint to get candidate tracks.

    Mapping logic:
      tempo           → target_tempo (direct)
      rms_mean        → target_energy (scaled)
      harmonic_ratio  → target_valence (proxy)
      beat_regularity → target_danceability (proxy)
      spectral_centroid → target_acousticness (inverse proxy)
      zcr_mean        → target_instrumentalness (inverse proxy)
    """
    tempo    = features.get('tempo', 120.0)
    rms_mean = features.get('rms_mean', 0.05)

    target_energy        = float(np.clip(rms_mean * 10 + (tempo - 60) / 200, 0, 1))
    target_valence       = float(np.clip(features.get('harmonic_ratio', 0.5) * 0.6
                                         + features.get('chroma_std', 0.1) * 2, 0, 1))
    target_dance         = float(np.clip(features.get('beat_regularity', 1.0) / 50.0, 0, 1))
    target_acoustic      = float(np.clip(1.0 - (features.get('spectral_centroid_mean', 2000)
                                                  / 8000), 0, 1))
    target_instrumental  = float(np.clip(1.0 - features.get('zcr_mean', 0.05) * 10, 0, 1))

    genre_seeds = ['pop', 'rock', 'electronic']

    try:
        recs = sp.recommendations(
            seed_genres=genre_seeds,
            limit=n_results,
            target_energy=round(target_energy, 3),
            target_valence=round(target_valence, 3),
            target_danceability=round(target_dance, 3),
            target_acousticness=round(target_acoustic, 3),
            target_instrumentalness=round(target_instrumental, 3),
            target_tempo=round(tempo, 1),
            min_popularity=20
        )
        tracks = recs.get('tracks', [])
    except Exception as e:
        st.warning(f"Spotify Recommendations error: {e}")
        tracks = []

    results = []
    for t in tracks:
        results.append({
            'id':           t['id'],
            'name':         t['name'],
            'artist':       ', '.join(a['name'] for a in t['artists']),
            'preview_url':  t.get('preview_url'),
            'external_url': t['external_urls'].get('spotify', ''),
            'album_art':    t['album']['images'][0]['url']
                            if t['album']['images'] else None,
            'popularity':   t.get('popularity', 0),
            # Store mapped targets on the track for similarity calc
            '_target_tempo':        tempo,
            '_target_energy':       target_energy,
            '_target_valence':      target_valence,
            '_target_dance':        target_dance,
            '_target_acoustic':     target_acoustic,
            '_target_instrumental': target_instrumental,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ML FEEDBACK MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """Build a sklearn GradientBoosting pipeline with standard scaling."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        ))
    ])


def train_model(feedback_data: list):
    """
    Train the feel-match predictor on collected user feedback.
    Returns trained pipeline, or None if insufficient data.
    """
    if len(feedback_data) < MIN_SAMPLES_TO_TRAIN:
        return None

    X = np.array([fv for fv, _ in feedback_data])
    y = np.array([r for _, r in feedback_data], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_model():
    """Load a previously saved model from disk."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_score(pipeline, feature_vec: np.ndarray) -> float:
    """Predict feel-match score (1–10) for a feature vector."""
    if pipeline is None:
        return 5.0
    fv = np.nan_to_num(feature_vec, nan=0.0).reshape(1, -1)
    score = pipeline.predict(fv)[0]
    return float(np.clip(score, 1.0, 10.0))


def get_feature_weights(pipeline, n_features: int) -> np.ndarray:
    """
    Extract normalised feature importances from the trained model.
    Used to re-weight cosine similarity — features you respond to
    most get amplified in future searches.
    """
    if pipeline is None:
        return np.ones(n_features, dtype=np.float32)
    importances = pipeline.named_steps['model'].feature_importances_
    weights = importances / (importances.mean() + 1e-10)
    return weights.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY & RANKING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def weighted_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray,
                                weights: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors with per-feature weights.
    Higher weight on a dimension = that audio property matters more.
    """
    wa = vec_a * weights
    wb = vec_b * weights
    if np.linalg.norm(wa) == 0 or np.linalg.norm(wb) == 0:
        return 0.0
    return float(1.0 - cosine(wa, wb))


def spotify_track_to_vector(track: dict, ref_features: dict) -> np.ndarray:
    """
    Build a proxy feature vector for a Spotify track (aligned to the
    reference clip's feature space) using the mapped Spotify attributes.
    Dimensions not covered are left as zero (neutral).
    """
    n = len(ref_features)
    vec = np.zeros(n, dtype=np.float32)
    feature_names = list(ref_features.keys())

    def set_feat(name, value):
        if name in feature_names:
            vec[feature_names.index(name)] = float(value)

    set_feat('tempo',                  track.get('_target_tempo', 120.0))
    set_feat('rms_mean',               track.get('_target_energy', 0.5) * 0.1)
    set_feat('harmonic_ratio',         track.get('_target_valence', 0.5) * 0.8)
    set_feat('beat_regularity',        track.get('_target_dance', 0.5) * 50)
    set_feat('spectral_centroid_mean', (1.0 - track.get('_target_acoustic', 0.5)) * 8000)
    set_feat('zcr_mean',               (1.0 - track.get('_target_instrumental', 0.5)) * 0.1)

    return vec


def rank_candidates(candidates: list, ref_feature_dict: dict,
                    ref_feature_vec: np.ndarray,
                    ml_pipeline, weights: np.ndarray) -> list:
    """
    Score every candidate track and return them sorted best → worst.

    Combined score formula:
        Score = 0.6 × WeightedCosineSim + 0.4 × (ML_Prediction / 10)
    """
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


def generate_session_id() -> str:
    return str(uuid.uuid4())[:8]


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Page Config ────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="MoodMatch 🎵",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Custom CSS ─────────────────────────────────────────────────────────────
    st.markdown("""
        <style>.main { background-color: #0e1117; }.stMetric { background-color: #1e2130; border-radius: 8px; padding: 10px; }.track-card {
            background-color: #1e2130;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }.score-badge {
            display: inline-block;
            background-color: #1db954;
            color: white;
            border-radius: 20px;
            padding: 2px 10px;
            font-size: 0.85em;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Initialise DB ──────────────────────────────────────────────────────────
    init_db()

    # ── Session State ──────────────────────────────────────────────────────────
    if 'session_id'      not in st.session_state:
        st.session_state.session_id      = generate_session_id()
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'ref_features'    not in st.session_state:
        st.session_state.ref_features    = None
    if 'ref_vec'         not in st.session_state:
        st.session_state.ref_vec         = None
    if 'feedback_given'  not in st.session_state:
        st.session_state.feedback_given  = {}
    if 'ml_pipeline'     not in st.session_state:
        st.session_state.ml_pipeline     = load_model()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎵 MoodMatch")
        st.caption("Find songs that make you *feel* the same way.")
        st.divider()

        st.subheader("🤖 ML Model Status")
        count = get_feedback_count()
        st.metric("Total ratings collected", count)

        if count >= MIN_SAMPLES_TO_TRAIN:
            st.success("✅ ML model active — results are personalised to you!")
        else:
            remaining = MIN_SAMPLES_TO_TRAIN - count
            st.info(f"Rate **{remaining}** more song(s) to activate ML personalisation.")

        st.progress(min(count / MIN_SAMPLES_TO_TRAIN, 1.0))

        if count >= MIN_SAMPLES_TO_TRAIN:
            if st.button("🔄 Retrain Model Now", use_container_width=True):
                with st.spinner("Training..."):
                    pipeline = train_model(load_all_feedback())
                    if pipeline:
                        st.session_state.ml_pipeline = pipeline
                        st.success("Model retrained!")

        st.divider()

        st.subheader("ℹ️ How It Works")
        st.markdown("""
        1. **Upload** any song file
        2. **Select** the 10–20s part that gives you *that feeling*
        3. We **analyse** rhythm, harmony, timbre, pitch, amplitude & more
        4. We **search Spotify** for emotional matches
        5. **Rate** each result → the ML model learns your taste
        """)

        st.divider()
        st.caption(f"Session: `{st.session_state.session_id}`")

    # ── Header ─────────────────────────────────────────────────────────────────
    st.title("🎵 MoodMatch")
    st.markdown(
        "#### Find songs that make you feel *exactly* the same way — "
        "matched by **rhythm, harmony, timbre, pitch & amplitude**, not just genre."
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — UPLOAD
    # ══════════════════════════════════════════════════════════════════════════
    st.header("🎤 Step 1 — Upload Your Song")
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
        help="Your file is processed locally — nothing is uploaded to any server."
    )

    if not uploaded_file:
        st.info("👆 Upload a song to get started.")
        st.stop()

    file_bytes = uploaded_file.read()
    st.audio(file_bytes, format=uploaded_file.type)
    st.success(f"✅ Loaded: **{uploaded_file.name}** "
               f"({len(file_bytes) / 1_000_000:.1f} MB)")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — SELECT SEGMENT
    # ══════════════════════════════════════════════════════════════════════════
    st.header("✂️ Step 2 — Select Your Favourite Part")
    st.markdown(
        "Pick the **10–20 second window** that captures the feeling you want to match. "
        "Be as precise as possible — this is your emotional fingerprint."
    )

    col1, col2 = st.columns(2)
    with col1:
        start_sec = st.number_input(
            "⏱ Start time (seconds)",
            min_value=0.0, max_value=3600.0,
            value=30.0, step=0.5,
            help="When does your favourite part begin?"
        )
    with col2:
        end_sec = st.number_input(
            "⏱ End time (seconds)",
            min_value=start_sec + 1.0, max_value=3600.0,
            value=start_sec + 15.0, step=0.5,
            help="When does it end? (10–20 seconds after start)"
        )

    duration = end_sec - start_sec

    if duration < 10.0:
        st.warning(f"⚠️ Segment too short ({duration:.1f}s). Minimum is 10 seconds.")
        st.stop()
    elif duration > 20.0:
        st.warning(f"⚠️ Segment too long ({duration:.1f}s). Maximum is 20 seconds.")
        st.stop()
    else:
        st.success(
            f"✅ Segment: **{start_sec:.1f}s → {end_sec:.1f}s** "
            f"({duration:.1f} seconds)"
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — ANALYSE & SEARCH
    # ══════════════════════════════════════════════════════════════════════════
    st.header("🔬 Step 3 — Analyse & Find Matches")

    if st.button("🚀 Analyse My Clip & Find Matches",
                 type="primary", use_container_width=True):

        # ── Audio Analysis ─────────────────────────────────────────────────
        with st.spinner("🎼 Extracting audio fingerprint from your clip..."):
            try:
                y, sr = load_audio_segment(file_bytes, start_sec, end_sec)
                features = extract_features(y, sr)
                feat_vec = features_to_vector(features)
                st.session_state.ref_features = features
                st.session_state.ref_vec      = feat_vec
            except Exception as e:
                st.error(f"❌ Audio analysis failed: {e}")
                st.stop()

        st.success("✅ Audio fingerprint extracted!")

        # ── Feature Summary ────────────────────────────────────────────────
        with st.expander("🔬 Your Clip's Audio Fingerprint", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🥁 Tempo",          f"{features['tempo']:.1f} BPM")
            c2.metric("⚡ Energy (RMS)",    f"{features['rms_mean']:.4f}")
            c3.metric("🎸 Harmonic Ratio",  f"{features['harmonic_ratio']:.1%}")
            c4.metric("🥁 Percussive Ratio",f"{features['percussive_ratio']:.1%}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("💓 Beat Strength",   f"{features['beat_strength_mean']:.3f}")
            c6.metric("🌈 Spectral Centre", f"{features['spectral_centroid_mean']:.0f} Hz")
            c7.metric("🎯 Beat Regularity", f"{features['beat_regularity']:.1f}")
            c8.metric("〰️ Texture (ZCR)",   f"{features['zcr_mean']:.4f}")

            st.markdown("**🎹 MFCC Timbre Fingerprint** (13 timbral coefficients):")
            mfcc_data = {f"MFCC {i}": features[f'mfcc_{i}_mean'] for i in range(13)}
            st.bar_chart(mfcc_data)

            st.markdown("**🎵 Chroma (Pitch Class Energy):**")
            note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            chroma_data = {note_names[i]: features[f'chroma_{i}'] for i in range(12)}
            st.bar_chart(chroma_data)

        # ── Spotify Search ─────────────────────────────────────────────────
        with st.spinner("🎵 Searching Spotify for emotional matches..."):
            try:
                sp = get_spotify_client()
                candidates = search_tracks_by_features(sp, features, n_results=50)
            except Exception as e:
                st.error(f"❌ Spotify API error: {e}")
                st.stop()

        if not candidates:
            st.warning("No candidates returned from Spotify. "
                       "Check your API credentials in the.env file.")
            st.stop()

        # ── Rank Candidates ────────────────────────────────────────────────
        pipeline = st.session_state.ml_pipeline
        weights  = get_feature_weights(pipeline, len(feat_vec))
        ranked   = rank_candidates(candidates, features, feat_vec, pipeline, weights)

        st.session_state.recommendations = ranked[:10]
        st.session_state.feedback_given  = {}
        st.success(f"✅ Found **{len(candidates)}** candidates — "
                   f"showing your top **10** emotional matches!")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — RESULTS & FEEDBACK
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.recommendations:
        st.divider()
        st.header("🎧 Step 4 — Your Emotional Matches")
        st.markdown(
            "Listen to each track preview. Then **rate how well it made you "
            "feel the same way** as your clip (1 = not at all → 10 = identical feeling). "
            "Your ratings teach the ML model your personal taste."
        )

        for i, track in enumerate(st.session_state.recommendations):

            already_rated = track['id'] in st.session_state.feedback_given

            with st.container():
                st.markdown(f"---")
                col_img, col_main, col_rate = st.columns([1, 5, 3])

                # Album art
                with col_img:
                    if track.get('album_art'):
                        st.image(track['album_art'], width=90)
                    else:
                        st.markdown("### 🎵")

                # Track info + preview
                with col_main:
                    rank_emoji = ["🥇","🥈","🥉","4️⃣","5️⃣",
                                  "6️⃣","7️⃣","8️⃣","9️⃣","🔟"][i]
                    st.markdown(
                        f"{rank_emoji} **[{track['name']}]({track['external_url']})**  "
                        f"— *{track['artist']}*"
                    )
                    st.caption(
                        f"🎯 Similarity: **{track['similarity_score']}%**  |  "
                        f"🤖 ML Prediction: **{track['ml_predicted_score']}/10**  |  "
                        f"⭐ Combined Score: **{track['combined_score']}%**"
                    )

                    if track.get('preview_url'):
                        st.audio(track['preview_url'], format='audio/mp3')
                    else:
                        st.caption("*No 30s preview available — "
                                   "[open on Spotify](%s) to listen*"
                                   % track['external_url'])

                # Rating
                with col_rate:
                    if already_rated:
                        given = st.session_state.feedback_given[track['id']]
                        bar   = "█" * given + "░" * (10 - given)
                        st.success(f"Rated: {given}/10\n`{bar}`")
                    else:
                        rating = st.slider(
                            "How did it feel?",
                            min_value=1, max_value=10, value=5,
                            key=f"slider_{track['id']}",
                            help="1 = totally different feeling | "
                                 "10 = exactly the same feeling"
                        )
                        if st.button("✅ Submit Rating",
                                     key=f"btn_{track['id']}",
                                     use_container_width=True):
                            save_feedback(
                                session_id=st.session_state.session_id,
                                track_id=track['id'],
                                track_name=track['name'],
                                artist=track['artist'],
                                rating=rating,
                                feature_vec=st.session_state.ref_vec
                            )
                            st.session_state.feedback_given[track['id']] = rating

                            # Auto-retrain every 5 new ratings
                            total = get_feedback_count()
                            if total >= MIN_SAMPLES_TO_TRAIN and total % 5 == 0:
                                new_pipeline = train_model(load_all_feedback())
                                if new_pipeline:
                                    st.session_state.ml_pipeline = new_pipeline
                                    st.toast("🤖 ML model updated with your feedback!")

                            st.rerun()

        # ── Session Summary ────────────────────────────────────────────────
        if st.session_state.feedback_given:
            st.divider()
            st.subheader("📊 Your Ratings This Session")

            rated = [(t, st.session_state.feedback_given[t['id']])
                     for t in st.session_state.recommendations
                     if t['id'] in st.session_state.feedback_given]
            rated.sort(key=lambda x: x[1], reverse=True)

            for track, r in rated:
                bar = "█" * r + "░" * (10 - r)
                col_a, col_b = st.columns([4, 1])
                col_a.markdown(f"**{track['name']}** — *{track['artist']}*  \n`{bar}`")
                col_b.metric("Score", f"{r}/10")

            avg = np.mean([r for _, r in rated])
            st.info(
                f"📈 Average feel-match score this session: **{avg:.1f}/10** "
                f"across {len(rated)} rated track(s). "
                f"Keep rating to improve ML accuracy!"
            )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()