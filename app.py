"""
app.py
------
CineMatch -- a content-based Movie Recommendation System built on the
CountVectorizer + cosine-similarity pipeline from the project notebook,
with posters and details fetched live from TMDB.

Run with:  streamlit run app.py
"""

import streamlit as st

from utils import (
    load_movie_data,
    get_recommendations,
    fetch_movie_details,
    movie_card_html,
    inject_css,
    TMDB_API_KEY,
)

st.set_page_config(
    page_title="CineMatch \u2014 Discover Your Next Favorite Film",
    page_icon="\U0001F3AC",
    layout="wide",
)

inject_css()


def render_header():
    st.markdown(
        """
        <div class="cm-header">
            <div class="cm-logo">Cine<span>Match</span></div>
            <div class="cm-tagline">Discover your next favorite film</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_details_page(movies, movie_id):
    st.markdown('<a class="cm-back-link" href="?" target="_self">\u2190 Back to Discover</a>',
                unsafe_allow_html=True)

    details = fetch_movie_details(movie_id)

    # Prefer the title from our own dataset if TMDB didn't return one,
    # so the page never shows a blank heading.
    fallback_row = movies[movies["movie_id"] == movie_id]
    display_title = details.get("title") or (
        fallback_row.iloc[0]["title"] if not fallback_row.empty else "Unknown Title"
    )

    if not details.get("ok"):
        st.info(details["overview"])

    col_poster, col_info = st.columns([1, 2], gap="large")

    with col_poster:
        st.markdown(
            f'<div class="cm-details-poster"><img src="{details["poster_url"]}" /></div>',
            unsafe_allow_html=True,
        )

    with col_info:
        st.markdown(f'<div class="cm-feature-title">{display_title}</div>', unsafe_allow_html=True)

        meta_bits = []
        if details.get("release_year"):
            meta_bits.append(details["release_year"])
        if details.get("rating"):
            meta_bits.append(f"\u2605 {details['rating']:.1f}/10")
        if meta_bits:
            meta_text = "  \u2022  ".join(meta_bits)
            st.markdown(
                f'<div class="cm-feature-year">{meta_text}</div>',
                unsafe_allow_html=True,
            )

        if details.get("genres"):
            badges = "".join(f'<span class="cm-badge">{g}</span>' for g in details["genres"])
            st.markdown(badges, unsafe_allow_html=True)

        st.markdown(f'<p class="cm-feature-overview">{details["overview"]}</p>', unsafe_allow_html=True)

        if details.get("director"):
            st.markdown(
                f'<p style="margin-top:1rem;">Directed by <span class="cm-director">{details["director"]}</span></p>',
                unsafe_allow_html=True,
            )

        if details.get("crew_highlights"):
            st.markdown(
                "<p style='color:#8f8272; font-size:0.85rem;'>" + " &nbsp;\u2022&nbsp; ".join(details["crew_highlights"]) + "</p>",
                unsafe_allow_html=True,
            )

        if details.get("cast"):
            st.markdown('<div class="cm-section-title">Cast</div>', unsafe_allow_html=True)
            for name in details["cast"]:
                st.markdown(f'<div class="cm-cast-name">{name}</div>', unsafe_allow_html=True)


def render_home_page(movies, similarity):
    st.markdown(
        """
        <div class="cm-hero">
            <h1>Find your next watch</h1>
            <p>Search for a movie you love, and we'll find five more you might enjoy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not TMDB_API_KEY:
        st.warning(
            "No TMDB API key detected. Posters and details will be limited until "
            "`TMDB_API_KEY` is set in your `.env` file.",
            icon="\u26A0\uFE0F",
        )

    titles = sorted(movies["title"].dropna().unique().tolist())

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        selected_title = st.selectbox(
            "Search for a movie",
            options=titles,
            index=0,
            key="movie_select",
            label_visibility="collapsed",
            placeholder="Start typing a movie title\u2026",
        )

    if not selected_title:
        return

    selected_row = movies[movies["title"] == selected_title].iloc[0]
    selected_id = selected_row["movie_id"]
    selected_details = fetch_movie_details(selected_id)

    st.markdown('<div class="cm-section-title">Selected movie</div>', unsafe_allow_html=True)

    with st.container(key="cm_feature_card"):
        col_poster, col_info = st.columns([1, 2.3], gap="large")

        with col_poster:
            st.markdown(
                f'<a href="?movie_id={selected_id}" target="_self">'
                f'<img src="{selected_details["poster_url"]}" /></a>',
                unsafe_allow_html=True,
            )

        with col_info:
            st.markdown(
                f'<a href="?movie_id={selected_id}" target="_self" style="text-decoration:none;">'
                f'<div class="cm-feature-title">{selected_title}</div></a>',
                unsafe_allow_html=True,
            )
            if selected_details.get("release_year"):
                st.markdown(
                    f'<div class="cm-feature-year">{selected_details["release_year"]}</div>',
                    unsafe_allow_html=True,
                )
            overview_text = selected_details["overview"]
            truncated_overview = overview_text[:320] + ("\u2026" if len(overview_text) > 320 else "")
            st.markdown(
                f'<p class="cm-feature-overview">{truncated_overview}</p>',
                unsafe_allow_html=True,
            )
            st.write("")
            recommend_clicked = st.button("\U0001F3AF Recommend Movies", key="recommend_btn")

    if recommend_clicked:
        st.session_state["recs_for"] = selected_title
        st.session_state["recs"] = get_recommendations(movies, similarity, selected_title, top_n=5)

    if st.session_state.get("recs_for") == selected_title:
        recs = st.session_state.get("recs", [])
        if recs:
            st.markdown(
                f'<div class="cm-section-title">Because you liked {selected_title}</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(5, gap="medium")
            for col, (rec_id, rec_title) in zip(cols, recs):
                rec_details = fetch_movie_details(rec_id)
                with col:
                    st.markdown(
                        movie_card_html(rec_id, rec_title, rec_details["poster_url"]),
                        unsafe_allow_html=True,
                    )
        else:
            st.info("We couldn't find similar movies for this title in the dataset.")


def main():
    render_header()

    movies, similarity = load_movie_data()

    query_movie_id = st.query_params.get("movie_id")

    if query_movie_id:
        try:
            movie_id = int(query_movie_id)
        except (TypeError, ValueError):
            movie_id = None

        if movie_id is not None:
            render_details_page(movies, movie_id)
        else:
            st.error("Invalid movie reference.")
            render_home_page(movies, similarity)
    else:
        render_home_page(movies, similarity)


if __name__ == "__main__":
    main()
