"""
utils.py
--------
Data loading, TMDB API integration, recommendation logic, and styling
helpers for the CineMatch movie recommendation app.

This mirrors the recommendation logic built in the project notebook:
- movie_list.pkl  -> DataFrame with columns [movie_id, title, tags]
- similarity.pkl  -> cosine similarity matrix aligned to the DataFrame's
                      (reset, contiguous) index
"""

import os
import base64
import pickle

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ---------------------------------------------------------------------------
# Placeholder poster (inline SVG data-URI) shown whenever a real poster
# can't be fetched -- keeps the UI clean instead of a broken image icon.
# ---------------------------------------------------------------------------
_PLACEHOLDER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="500" height="750">
  <rect width="100%" height="100%" fill="#241c17"/>
  <rect x="18" y="18" width="464" height="714" fill="none" stroke="#c9975b" stroke-width="2"/>
  <text x="50%" y="47%" font-family="Georgia, serif" font-size="26" fill="#c9975b" text-anchor="middle">No Poster</text>
  <text x="50%" y="54%" font-family="Georgia, serif" font-size="26" fill="#c9975b" text-anchor="middle">Available</text>
</svg>
""".strip()
PLACEHOLDER_POSTER = "data:image/svg+xml;base64," + base64.b64encode(
    _PLACEHOLDER_SVG.encode()
).decode()


# ---------------------------------------------------------------------------
# Data loading (cached so the pickle files are only read once per session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_movie_data():
    """
    Loads the preprocessed movie dataframe and cosine similarity matrix
    produced by the project notebook.
    """
    try:
        movies = pickle.load(open("movie_list.pkl", "rb"))
        similarity = pickle.load(open("similarity.pkl", "rb"))
    except FileNotFoundError:
        st.error(
            "Couldn't find `movie_list.pkl` / `similarity.pkl` in the app folder. "
            "Run the project notebook first (through the final pickle.dump cell) "
            "and place the two files alongside `app.py`."
        )
        st.stop()

    # Defensive: make sure the dataframe has a clean contiguous index so it
    # stays aligned with the similarity matrix's row/column positions.
    movies = movies.reset_index(drop=True)
    return movies, similarity


# ---------------------------------------------------------------------------
# Recommendation logic (adapted directly from the notebook's recommend())
# ---------------------------------------------------------------------------
def get_recommendations(movies: pd.DataFrame, similarity, title: str, top_n: int = 5):
    """
    Returns up to `top_n` (movie_id, title) tuples most similar to `title`,
    excluding the movie itself. Same idea as the notebook's recommend(),
    but returns data instead of printing it, and guards against a title
    that isn't found (or duplicate titles, using the first match).
    """
    matches = movies.index[movies["title"] == title]
    if len(matches) == 0:
        return []

    index = matches[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1]
    )

    recommendations = []
    for i, _score in distances[1: top_n + 1]:
        row = movies.iloc[i]
        recommendations.append((row["movie_id"], row["title"]))
    return recommendations


# ---------------------------------------------------------------------------
# TMDB API integration
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_movie_details(movie_id):
    """
    Fetches poster, overview, genres, release year, rating, cast and
    director for a TMDB movie id in a single request (append_to_response).
    Always returns a usable dict -- never raises -- so the UI can degrade
    gracefully on any API/network problem.
    """
    fallback = {
        "title": None,
        "poster_url": PLACEHOLDER_POSTER,
        "overview": "We couldn't retrieve details for this movie right now.",
        "release_year": None,
        "genres": [],
        "rating": None,
        "cast": [],
        "director": None,
        "crew_highlights": [],
        "ok": False,
    }

    if not movie_id or pd.isna(movie_id):
        return fallback

    if not TMDB_API_KEY:
        fallback["overview"] = (
            "Set TMDB_API_KEY in your .env file to load posters and movie details."
        )
        return fallback

    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{int(movie_id)}",
            params={"api_key": TMDB_API_KEY, "append_to_response": "credits"},
            timeout=8,
        )
        if response.status_code != 200:
            return fallback
        data = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return fallback

    poster_path = data.get("poster_path")
    poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else PLACEHOLDER_POSTER

    release_date = data.get("release_date") or ""
    release_year = release_date.split("-")[0] if release_date else None

    genres = [g["name"] for g in (data.get("genres") or [])]

    credits = data.get("credits") or {}
    cast_list = credits.get("cast") or []
    top_cast = [c["name"] for c in cast_list[:5] if c.get("name")]

    crew_list = credits.get("crew") or []
    director = next(
        (c["name"] for c in crew_list if c.get("job") == "Director"), None
    )
    crew_highlights = [
        f"{c['name']} \u2014 {c['job']}"
        for c in crew_list
        if c.get("job") in ("Screenplay", "Writer", "Producer", "Director of Photography")
    ][:4]

    return {
        "title": data.get("title"),
        "poster_url": poster_url,
        "overview": data.get("overview") or "No overview available for this title.",
        "release_year": release_year,
        "genres": genres,
        "rating": data.get("vote_average"),
        "cast": top_cast,
        "director": director,
        "crew_highlights": crew_highlights,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------
def movie_card_html(movie_id, title, poster_url, meta=""):
    """A clickable poster card that navigates to the details view via a
    query param link -- works natively in Streamlit without extra JS."""
    meta_html = f'<div class="mc-meta">{meta}</div>' if meta else ""
    safe_title = title.replace('"', "&quot;")
    return f"""
    <a href="?movie_id={movie_id}" target="_self" class="movie-card">
        <div class="mc-poster">
            <img src="{poster_url}" alt="{safe_title}" loading="lazy" />
        </div>
        <div class="mc-title">{safe_title}</div>
        {meta_html}
    </a>
    """


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 20% 0%, #241c17 0%, #17110d 55%, #120d0a 100%);
            color: #f2e9dc;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1150px;
        }

        /* ---------- Header ---------- */
        .cm-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            border-bottom: 1px solid rgba(201, 151, 91, 0.25);
            padding-bottom: 0.9rem;
            margin-bottom: 2.2rem;
        }
        .cm-logo {
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: #f2e9dc;
            letter-spacing: 0.5px;
        }
        .cm-logo span { color: #c9975b; }
        .cm-tagline {
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #c9975b;
            border-left: 2px solid rgba(201, 151, 91, 0.5);
            padding-left: 0.6rem;
        }

        /* ---------- Hero ---------- */
        .cm-hero { text-align: center; margin-bottom: 1.8rem; }
        .cm-hero h1 {
            font-family: 'Playfair Display', serif;
            font-size: 2.3rem;
            font-weight: 700;
            color: #f6ede0;
            margin-bottom: 0.3rem;
        }
        .cm-hero p {
            color: #b3a596;
            font-size: 1rem;
            margin-top: 0;
        }

        /* ---------- Section headings ---------- */
        .cm-section-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.3rem;
            color: #f2e9dc;
            margin: 1.6rem 0 1rem 0;
            border-left: 3px solid #c9975b;
            padding-left: 0.6rem;
        }

        /* ---------- Selected movie feature card ---------- */
        div[class*="st-key-cm_feature_card"] {
            background: linear-gradient(145deg, #241c17, #1c1512);
            border: 1px solid rgba(201, 151, 91, 0.2);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }
        div[class*="st-key-cm_feature_card"] img {
            border-radius: 10px;
            width: 100%;
            box-shadow: 0 8px 20px rgba(0,0,0,0.45);
        }
        .cm-feature-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            color: #f6ede0;
            margin-bottom: 0.15rem;
        }
        .cm-feature-year {
            color: #c9975b;
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }
        .cm-feature-overview {
            color: #cfc2b2;
            line-height: 1.55;
            font-size: 0.95rem;
        }
        .cm-badge {
            display: inline-block;
            background: rgba(201, 151, 91, 0.15);
            color: #d9b481;
            border: 1px solid rgba(201, 151, 91, 0.35);
            border-radius: 20px;
            padding: 0.15rem 0.7rem;
            font-size: 0.75rem;
            margin: 0 0.35rem 0.35rem 0;
        }

        /* ---------- Movie cards (recommendations grid) ---------- */
        .movie-card {
            display: block;
            text-decoration: none;
            background: #1c1512;
            border: 1px solid rgba(201, 151, 91, 0.15);
            border-radius: 12px;
            padding: 0.6rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
            height: 100%;
        }
        .movie-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.4);
            border-color: rgba(201, 151, 91, 0.55);
        }
        .mc-poster img {
            width: 100%;
            border-radius: 8px;
            display: block;
            aspect-ratio: 2 / 3;
            object-fit: cover;
        }
        .mc-title {
            color: #f2e9dc;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.55rem;
            line-height: 1.25;
        }
        .mc-meta {
            color: #8f8272;
            font-size: 0.78rem;
            margin-top: 0.15rem;
        }

        /* ---------- Details page ---------- */
        .cm-details-poster img {
            width: 100%;
            border-radius: 12px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.5);
        }
        .cm-back-link {
            display: inline-block;
            color: #c9975b !important;
            text-decoration: none;
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(201, 151, 91, 0.3);
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
        }
        .cm-back-link:hover { background: rgba(201, 151, 91, 0.12); }

        .cm-cast-name {
            color: #e7dccb;
            font-size: 0.92rem;
            padding: 0.15rem 0;
        }
        .cm-director {
            color: #d9b481;
            font-weight: 600;
        }

        /* ---------- Streamlit widget theming ---------- */
        div[data-baseweb="select"] > div {
            background-color: #1c1512;
            border-color: rgba(201, 151, 91, 0.35);
            border-radius: 10px;
        }
        .stButton > button {
            background: linear-gradient(135deg, #c9975b, #a8763b);
            color: #16110c;
            border: none;
            border-radius: 24px;
            padding: 0.55rem 1.6rem;
            font-weight: 600;
            font-size: 0.95rem;
            box-shadow: 0 6px 16px rgba(201, 151, 91, 0.25);
            transition: transform 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            color: #16110c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
