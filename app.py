import os
import re
import sqlite3
from pathlib import Path
from html import escape

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "ml-latest-small"
DB_PATH = BASE_DIR / "cinematch.db"

PLACEHOLDER = (
    "https://placehold.co/500x750/181818/eeeeee"
    "?text=Poster+Unavailable"
)

TMDB_IMAGE = "https://image.tmdb.org/t/p/w500"


# =========================================================
# 2. NETFLIX-INSPIRED DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #141414;
        color: #f5f5f1;
    }

    [data-testid="stHeader"] {
        background: rgba(20, 20, 20, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #101010;
        border-right: 1px solid #303030;
    }

    [data-testid="stSidebar"] * {
        color: #eeeeee;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, p {
        color: #f5f5f1;
    }

    .brand {
        color: #e50914;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 2px;
    }

    .brand-subtitle {
        color: #aaaaaa;
        font-size: 10px;
        letter-spacing: 3px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 800;
        margin-top: 22px;
        margin-bottom: 8px;
    }

    .muted {
        color: #b3b3b3;
        font-size: 13px;
    }

    .hero {
        min-height: 340px;
        border-radius: 10px;
        padding: 35px;
        margin-top: 18px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 1px solid #303030;
        background-color: #252525;
        background-position: center 30%;
        background-size: cover;
    }

    .hero-kicker {
        color: #e50914;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 3px;
        margin-bottom: 10px;
    }

    .hero-title {
        color: white;
        font-size: clamp(30px, 4vw, 48px);
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    .hero-meta {
        color: #eeeeee;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .hero-description {
        color: #eeeeee;
        max-width: 560px;
        font-size: 14px;
        line-height: 1.7;
    }

    .movie-title {
        color: #ffffff;
        font-size: 14px;
        font-weight: 750;
        margin-top: 8px;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .movie-meta {
        color: #b3b3b3;
        font-size: 11px;
        line-height: 1.6;
        margin-top: 3px;
    }

    .movie-rating {
        color: #f5c518;
        font-size: 12px;
        margin-top: 5px;
    }

    [data-testid="stImage"] img {
        border-radius: 6px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #e50914;
        color: white;
        border: 1px solid #e50914;
        border-radius: 5px;
        font-weight: 700;
        min-height: 38px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #b20710;
        color: white;
        border-color: #b20710;
    }

    div[data-testid="stMetric"] {
        background: #202020;
        border: 1px solid #353535;
        border-radius: 8px;
        padding: 14px;
    }

    div[data-testid="stForm"] {
        background: #1c1c1c;
        border: 1px solid #303030;
        padding: 18px;
        border-radius: 9px;
    }

    [data-testid="stExpander"] {
        background: #1c1c1c;
        border-color: #333333;
    }

    hr {
        border-color: #333333;
    }

    .footer {
        color: #777777;
        font-size: 11px;
        text-align: center;
        padding: 28px 0;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 20px;
            min-height: 270px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD MOVIE DATA AND BUILD AI ENGINE
# =========================================================

@st.cache_resource(show_spinner="Loading CineMatch movie library...")
def load_movie_engine():
    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"
    tags_path = DATA_DIR / "tags.csv"

    for path in (movies_path, ratings_path, tags_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Check the data/ml-latest-small folder."
            )

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)

    stats = (
        ratings.groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        )
        .reset_index()
    )

    tag_text = (
        tags.dropna(subset=["tag"])
        .groupby("movieId")["tag"]
        .apply(lambda values: " ".join(values.astype(str)))
        .rename("tags")
        .reset_index()
    )

    movies = movies.merge(stats, on="movieId", how="left")
    movies = movies.merge(tag_text, on="movieId", how="left")

    movies["avg_rating"] = movies["avg_rating"].fillna(0.0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    movies["genres"] = movies["genres"].fillna("")
    movies["tags"] = movies["tags"].fillna("")

    movies["year"] = movies["title"].str.extract(
        r"\((\d{4})\)\s*$"
    )

    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$",
        "",
        regex=True,
    )

    genre_text = movies["genres"].str.replace(
        "|",
        " ",
        regex=False,
    )

    movies["features"] = (
        movies["clean_title"]
        + " "
        + genre_text
        + " "
        + genre_text
        + " "
        + movies["tags"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(movies["features"])

    movies = movies.reset_index(drop=True)

    return movies, matrix


# =========================================================
# 4. TMDB POSTERS
# =========================================================

def get_api_key():
    try:
        return st.secrets.get(
            "TMDB_API_KEY",
            os.getenv("TMDB_API_KEY", ""),
        )
    except Exception:
        return os.getenv("TMDB_API_KEY", "")


@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(title, year, api_key):
    if not api_key:
        return PLACEHOLDER

    clean_title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        str(title),
    )

    params = {
        "api_key": api_key,
        "query": clean_title,
        "include_adult": "false",
    }

    if pd.notna(year) and str(year).strip():
        params["year"] = str(year)

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=8,
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        if results and results[0].get("poster_path"):
            return TMDB_IMAGE + results[0]["poster_path"]

    except (requests.RequestException, ValueError):
        pass

    return PLACEHOLDER


def movie_poster(movie, api_key):
    return get_poster(
        str(movie["title"]),
        movie["year"],
        api_key,
    )


def movie_year(movie):
    year = movie.get("year")

    if pd.notna(year) and str(year).strip():
        return str(int(float(year)))

    return "Year unknown"


def movie_genres(movie):
    genres = str(movie.get("genres", ""))

    if not genres or genres == "(no genres listed)":
        return "Uncategorized"

    return genres.replace("|", " · ")


# =========================================================
# 5. SQLITE DATABASE
# =========================================================

def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id INTEGER NOT NULL,
                reviewer TEXT NOT NULL,
                review TEXT NOT NULL,
                rating INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                movie_id INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, movie_id)
            )
        """)


def save_review(movie_id, reviewer, review, rating):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO reviews
                (movie_id, reviewer, review, rating)
            VALUES (?, ?, ?, ?)
            """,
            (
                int(movie_id),
                reviewer.strip(),
                review.strip(),
                int(rating),
            ),
        )


def get_reviews(movie_id):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT reviewer, review, rating, created_at
            FROM reviews
            WHERE movie_id = ?
            ORDER BY id DESC
            """,
            conn,
            params=(int(movie_id),),
        )


def add_to_watchlist(username, movie_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO watchlist (username, movie_id)
            VALUES (?, ?)
            """,
            (username, int(movie_id)),
        )


def remove_from_watchlist(username, movie_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            DELETE FROM watchlist
            WHERE username = ? AND movie_id = ?
            """,
            (username, int(movie_id)),
        )


def get_watchlist_ids(username):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT movie_id
            FROM watchlist
            WHERE username = ?
            ORDER BY id DESC
            """,
            (username,),
        ).fetchall()

    return [int(row[0]) for row in rows]


# =========================================================
# 6. INITIALIZE APP AND SESSION STATE
# =========================================================

try:
    movies, tfidf_matrix = load_movie_engine()
    initialize_database()

except Exception as error:
    st.error(f"Could not start CineMatch: {error}")
    st.info(
        "Verify that movies.csv, ratings.csv, and tags.csv "
        "exist in data/ml-latest-small/."
    )
    st.stop()

api_key = get_api_key()

movie_id_to_index = {
    int(movie_id): idx
    for idx, movie_id in enumerate(movies["movieId"])
}

movie_titles = sorted(
    movies["title"].dropna().unique().tolist()
)

available_genres = sorted({
    genre
    for value in movies["genres"]
    for genre in str(value).split("|")
    if genre and genre != "(no genres listed)"
})

if "navigation" not in st.session_state:
    st.session_state.navigation = "Home"

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

if "watchlist_user" not in st.session_state:
    st.session_state.watchlist_user = "Guest"

if "movie_picker" not in st.session_state:
    st.session_state.movie_picker = (
        "Toy Story (1995)"
        if "Toy Story (1995)" in movie_titles
        else movie_titles[0]
    )

if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = st.session_state.navigation


# =========================================================
# 7. NAVIGATION CALLBACK
# =========================================================

def sync_navigation():
    st.session_state.navigation = st.session_state.nav_radio


def open_movie(movie_id):
    st.session_state.selected_movie_id = int(movie_id)
    st.session_state.navigation = "Movie details"


def open_page(page_name):
    st.session_state.navigation = page_name


# =========================================================
# 8. RECOMMENDATION ENGINE
# =========================================================

@st.cache_data(max_entries=512, show_spinner=False)
def calculate_similarities(movie_index):
    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix,
    ).flatten()

    return scores.tolist()


def get_similar_movies(movie_index, limit=12, genre="All genres"):
    scores = calculate_similarities(movie_index)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )

    results = []

    for idx in ranked_indices:
        if idx == movie_index:
            continue

        movie = movies.iloc[idx]

        if genre != "All genres":
            if genre not in str(movie["genres"]).split("|"):
                continue

        results.append((idx, scores[idx]))

        if len(results) >= limit:
            break

    return results


# =========================================================
# 9. REUSABLE MOVIE SHELF
# =========================================================

def section_heading(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{escape(title)}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.caption(subtitle)


def show_movie_shelf(
    title,
    frame,
    api_key,
    limit=10,
    key_prefix="shelf",
):
    section_heading(title)

    if frame.empty:
        st.caption("No movies found in this section.")
        return

    frame = frame.head(limit)

    # Five posters per row on wider screens.
    for batch_start in range(0, len(frame), 5):
        batch = frame.iloc[batch_start:batch_start + 5]
        columns = st.columns(5)

        for col, (_, movie) in zip(columns, batch.iterrows()):
            with col:
                st.image(
                    movie_poster(movie, api_key),
                    width="stretch",

                )

                st.markdown(
                    f'<div class="movie-title">'
                    f'{escape(str(movie["clean_title"]))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="movie-meta">'
                    f'{escape(movie_year(movie))} · '
                    f'{escape(movie_genres(movie))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if int(movie["rating_count"]) > 0:
                    st.markdown(
                        f'<div class="movie-rating">'
                        f'★ {float(movie["avg_rating"]):.1f}/5'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                if st.button(
                    "More info",
                    key=f"{key_prefix}_{int(movie['movieId'])}",
                    width="stretch"
,
                ):
                    open_movie(movie["movieId"])
                    st.rerun()


# =========================================================
# 10. MOVIE DETAIL PAGE
# =========================================================

def render_movie_details(movie, username):
    movie_index = movie_id_to_index.get(int(movie["movieId"]))

    if movie_index is None:
        st.error("Movie not found.")
        return

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.image(
            movie_poster(movie, api_key),
            width="stretch"
,
        )

    with right:
        st.markdown(
            '<div class="hero-kicker">CINEMATCH MOVIE GUIDE</div>',
            unsafe_allow_html=True,
        )

        st.title(str(movie["clean_title"]))

        st.caption(
            f"{movie_year(movie)} · {movie_genres(movie)}"
        )

        if int(movie["rating_count"]) > 0:
            st.write(
                f"⭐ **{float(movie['avg_rating']):.1f}/5** "
                f"from {int(movie['rating_count']):,} ratings"
            )
        else:
            st.write("No MovieLens ratings available.")

        st.write(
            "Explore related films selected by CineMatch's "
            "content-based recommendation engine."
        )

        saved_ids = get_watchlist_ids(username)

        if int(movie["movieId"]) in saved_ids:
            if st.button(
                "✓ Remove from My List",
                key="detail_remove",
                width="stretch"
,
            ):
                remove_from_watchlist(
                    username,
                    movie["movieId"],
                )
                st.rerun()

        else:
            if st.button(
                "+ Add to My List",
                key="detail_add",
                width="stretch"
,
            ):
                add_to_watchlist(
                    username,
                    movie["movieId"],
                )
                st.success("Added to My List.")
                st.rerun()

    st.divider()

    similar = get_similar_movies(movie_index, limit=10)

    similar_frame = movies.iloc[
        [idx for idx, _ in similar]
    ]

    show_movie_shelf(
        "More Like This",
        similar_frame,
        api_key,
        limit=10,
        key_prefix="detail_similar",
    )

    st.divider()

    section_heading(
        "Ratings & Reviews",
        f"Write a review for {movie['clean_title']}.",
    )

    with st.form(
        f"review_form_{int(movie['movieId'])}",
        clear_on_submit=True,
    ):
        personal_rating = st.slider(
            "Your rating",
            min_value=1,
            max_value=5,
            value=4,
            key=f"personal_rating_{int(movie['movieId'])}",
        )

        review_text = st.text_area(
            "Your review",
            placeholder="What did you think of this movie?",
            max_chars=1000,
            key=f"review_text_{int(movie['movieId'])}",
        )

        submitted = st.form_submit_button(
            "Submit review",
            width="stretch",
        )

        if submitted:
            if not review_text.strip():
                st.warning("Please enter a review first.")
            else:
                save_review(
                    movie["movieId"],
                    username,
                    review_text,
                    personal_rating,
                )
                st.success("Review saved.")
                st.rerun()

    reviews = get_reviews(movie["movieId"])

    if reviews.empty:
        st.caption("No written reviews yet. Be the first!")
    else:
        st.metric(
            "Community review average",
            f"{reviews['rating'].mean():.1f}/5",
            f"{len(reviews)} written reviews",
        )

        for _, review in reviews.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{escape(str(review['reviewer']))}** "
                    f"· ⭐ {int(review['rating'])}/5"
                )
                st.write(str(review["review"]))
                st.caption(str(review["created_at"]))


# =========================================================
# 11. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">CINEMATCH</div>'
        '<div class="brand-subtitle">MOVIE DISCOVERY</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # Important: synchronize the radio state BEFORE
    # instantiating the widget. This avoids Streamlit's
    # WidgetAlreadyInstantiatedError.
    valid_pages = [
        "Home",
        "Discover",
        "My List",
        "Movie details",
    ]

    if st.session_state.navigation not in valid_pages:
        st.session_state.navigation = "Home"

    if st.session_state.get("nav_radio") != st.session_state.navigation:
        st.session_state.nav_radio = st.session_state.navigation

    st.radio(
        "MENU",
        valid_pages,
        key="nav_radio",
        on_change=sync_navigation,
    )

    st.divider()

    st.markdown("### Your profile")

    username = st.text_input(
        "Display name",
        value=st.session_state.watchlist_user,
        max_chars=40,
        key="profile_name",
    ).strip()

    if not username:
        username = "Guest"

    st.session_state.watchlist_user = username

    st.divider()

    st.markdown("### Browse preferences")

    selected_genre = st.selectbox(
        "Filter by genre",
        ["All genres"] + available_genres,
        key="global_genre",
    )

    min_rating = st.slider(
        "Minimum rating",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        key="global_min_rating",
    )

    min_votes = st.slider(
        "Minimum rating count",
        min_value=0,
        max_value=500,
        value=0,
        step=25,
        key="global_min_votes",
    )

    if not api_key:
        st.caption(
            "TMDB API key not configured. "
            "Posters will use placeholders."
        )


# =========================================================
# 12. HEADER
# =========================================================

header_left, header_right = st.columns([1, 3])

with header_left:
    st.markdown(
        '<div class="brand">CINEMATCH</div>'
        '<div class="brand-subtitle">DISCOVER YOUR NEXT FILM</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        "<div style='text-align:right;color:#bbbbbb;"
        "padding-top:14px;font-size:13px;'>"
        "Movies · Recommendations · My List"
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# 13. SHARED FILTERING
# =========================================================

filtered_movies = movies.copy()

if selected_genre != "All genres":
    filtered_movies = filtered_movies[
        filtered_movies["genres"].apply(
            lambda value: selected_genre in str(value).split("|")
        )
    ]

filtered_movies = filtered_movies[
    (filtered_movies["avg_rating"] >= min_rating)
    & (filtered_movies["rating_count"] >= min_votes)
]


# =========================================================
# 14. HOME PAGE
# =========================================================

if st.session_state.navigation == "Home":

    # -----------------------------------------------------
    # Original movie selector: searchable dropdown with the
    # entire MovieLens movie catalog, not a small text box.
    # Streamlit's selectbox supports typing to search.
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🔎 Find a Movie You Like</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Browse the full movie list or start typing a title "
        "to jump straight to it."
    )

    default_title = (
        "Toy Story (1995)"
        if "Toy Story (1995)" in movie_titles
        else movie_titles[0]
    )

    if st.session_state.movie_picker not in movie_titles:
        st.session_state.movie_picker = default_title

    selected_title = st.selectbox(
        "Search or select a movie",
        options=movie_titles,
        index=movie_titles.index(st.session_state.movie_picker),
        key="movie_picker",
        help=(
            "Open the dropdown to browse movies, "
            "or type a title to search."
        ),
    )

    selected_index = movies.index[
        movies["title"] == selected_title
    ][0]

    selected_movie = movies.iloc[selected_index]

    # Hero movie selected through the original dropdown.
    hero_poster = movie_poster(selected_movie, api_key)

    hero_background = (
        "linear-gradient(90deg, rgba(0,0,0,.97) 0%, "
        "rgba(0,0,0,.78) 48%, rgba(0,0,0,.18) 100%), "
        f"url('{hero_poster}')"
    )

    st.markdown(
        f"""
        <div class="hero" style="
            background-image: {hero_background};
        ">
            <div class="hero-kicker">YOUR SELECTED MOVIE</div>
            <div class="hero-title">
                {escape(str(selected_movie["clean_title"]))}
            </div>
            <div class="hero-meta">
                {escape(movie_year(selected_movie))}
                &nbsp; · &nbsp;
                {escape(movie_genres(selected_movie))}
                &nbsp; · &nbsp;
                {"★ " + format(float(selected_movie["avg_rating"]), ".1f") + "/5"
                 if int(selected_movie["rating_count"]) > 0
                 else "Not rated"}
            </div>
            <div class="hero-description">
                Discover films like this one, explore your favorite genres,
                and build your own movie watchlist.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    detail_col, list_col = st.columns(2)

    with detail_col:
        if st.button(
            "▶ Explore This Movie",
            width="stretch",
        ):
            open_movie(selected_movie["movieId"])
            st.rerun()

    with list_col:
        current_watchlist = get_watchlist_ids(username)

        if int(selected_movie["movieId"]) in current_watchlist:
            if st.button(
                "✓ Remove from My List",
                key="home_remove",
                width="stretch",
            ):
                remove_from_watchlist(
                    username,
                    selected_movie["movieId"],
                )
                st.rerun()
        else:
            if st.button(
                "+ Add to My List",
                key="home_add",
                width="stretch",
            ):
                add_to_watchlist(
                    username,
                    selected_movie["movieId"],
                )
                st.success("Added to My List.")
                st.rerun()

    # -----------------------------------------------------
    # AI recommendations for the selected movie
    # -----------------------------------------------------

    section_heading(
        "✨ Recommended for You",
        "Movies ranked by similarity to the title you selected above.",
    )

    recommendations = get_similar_movies(
        selected_index,
        limit=10,
        genre=selected_genre,
    )

    if recommendations:
        recommendation_frame = movies.iloc[
            [idx for idx, _ in recommendations]
        ]

        show_movie_shelf(
            "Because You Like This Movie",
            recommendation_frame,
            api_key,
            limit=10,
            key_prefix="home_ai",
        )
    else:
        st.info(
            "No similar movies match this genre. "
            "Try choosing All genres in the sidebar."
        )

    st.divider()

    # -----------------------------------------------------
    # Netflix-style shelves
    # -----------------------------------------------------

    trending = movies[
        movies["rating_count"] >= 25
    ].sort_values(
        ["rating_count", "avg_rating"],
        ascending=[False, False],
    )

    top_rated = movies[
        movies["rating_count"] >= 50
    ].sort_values(
        ["avg_rating", "rating_count"],
        ascending=[False, False],
    )

    most_rated = movies.sort_values(
        "rating_count",
        ascending=False,
    )

    recent_movies = movies[
        movies["year"].notna()
    ].sort_values(
        "year",
        ascending=False,
    )

    genre_movies = filtered_movies.sort_values(
        ["rating_count", "avg_rating"],
        ascending=[False, False],
    )

    saved_ids = get_watchlist_ids(username)

    saved_movies = movies[
        movies["movieId"].isin(saved_ids)
    ]

    show_movie_shelf(
        "🔥 Popular Movies",
        trending,
        api_key,
        limit=10,
        key_prefix="home_trending",
    )

    show_movie_shelf(
        "⭐ Top Rated Movies",
        top_rated,
        api_key,
        limit=10,
        key_prefix="home_top",
    )

    show_movie_shelf(
        "🎬 Most Watched by MovieLens Users",
        most_rated,
        api_key,
        limit=10,
        key_prefix="home_watched",
    )

    show_movie_shelf(
        f"🎭 {selected_genre if selected_genre != 'All genres' else 'Explore All Genres'}",
        genre_movies,
        api_key,
        limit=10,
        key_prefix="home_genre",
    )

    show_movie_shelf(
        "🆕 Newer Movies in the Catalog",
        recent_movies,
        api_key,
        limit=10,
        key_prefix="home_recent",
    )

    show_movie_shelf(
        "❤️ My List",
        saved_movies,
        api_key,
        limit=10,
        key_prefix="home_saved",
    )


# =========================================================
# 15. DISCOVER PAGE
# =========================================================

elif st.session_state.navigation == "Discover":

    st.title("Discover Movies")
    st.caption(
        "Browse the catalog, filter by genre, or search for a title."
    )

    # This is also a full searchable movie dropdown.
    discover_title = st.selectbox(
        "Jump to a movie",
        options=["Choose a movie..."] + movie_titles,
        key="discover_movie_picker",
    )

    if discover_title != "Choose a movie...":
        if st.button(
            "Open Selected Movie",
            width="stretch",
        ):
            movie_id = int(
                movies.loc[
                    movies["title"] == discover_title,
                    "movieId",
                ].iloc[0]
            )
            open_movie(movie_id)
            st.rerun()

    st.divider()

    discover_query = st.text_input(
        "Filter movie results",
        placeholder="Type part of a movie title...",
        key="discover_query",
    )

    discover_genre = st.selectbox(
        "Genre",
        ["All genres"] + available_genres,
        key="discover_genre",
    )

    left, right = st.columns(2)

    with left:
        discover_rating = st.slider(
            "Minimum rating",
            0.0,
            5.0,
            0.0,
            0.5,
            key="discover_rating",
        )

    with right:
        discover_votes = st.slider(
            "Minimum rating count",
            0,
            500,
            0,
            25,
            key="discover_votes",
        )

    discover = movies.copy()

    if discover_query.strip():
        discover = discover[
            discover["clean_title"].str.contains(
                discover_query.strip(),
                case=False,
                regex=False,
                na=False,
            )
        ]

    if discover_genre != "All genres":
        discover = discover[
            discover["genres"].apply(
                lambda value: discover_genre in str(value).split("|")
            )
        ]

    discover = discover[
        (discover["avg_rating"] >= discover_rating)
        & (discover["rating_count"] >= discover_votes)
    ].sort_values(
        ["rating_count", "avg_rating"],
        ascending=[False, False],
    )

    st.write(f"**{len(discover):,} movies found**")

    if discover.empty:
        st.info("No movies match your filters.")
    else:
        page_size = 20

        total_pages = max(
            1,
            (len(discover) + page_size - 1) // page_size,
        )

        current_page = st.number_input(
            "Results page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key="discover_page_number",
        )

        start = (int(current_page) - 1) * page_size
        current_movies = discover.iloc[start:start + page_size]

        for batch_start in range(0, len(current_movies), 5):
            batch = current_movies.iloc[batch_start:batch_start + 5]
            columns = st.columns(5)

            for col, (_, movie) in zip(columns, batch.iterrows()):
                with col:
                    st.image(
                        movie_poster(movie, api_key),
                        width="stretch",
                    )

                    st.markdown(
                        f"**{escape(str(movie['clean_title']))}**"
                    )

                    st.caption(
                        f"{movie_year(movie)} · {movie_genres(movie)}"
                    )

                    if int(movie["rating_count"]) > 0:
                        st.caption(
                            f"⭐ {float(movie['avg_rating']):.1f}/5"
                        )

                    if st.button(
                        "More info",
                        key=f"discover_detail_{int(movie['movieId'])}",
                        width="stretch",
                    ):
                        open_movie(movie["movieId"])
                        st.rerun()


# =========================================================
# 16. MY LIST PAGE
# =========================================================

elif st.session_state.navigation == "My List":

    st.title("My List")
    st.caption(
        f"Your saved movies, {username}."
    )

    saved_ids = get_watchlist_ids(username)

    saved_movies = movies[
        movies["movieId"].isin(saved_ids)
    ]

    if saved_movies.empty:
        st.info(
            "Your list is empty. Go to Home, choose a movie, "
            "and select Add to My List."
        )

        if st.button("Browse Movies"):
            open_page("Home")
            st.rerun()

    else:
        st.write(f"**{len(saved_movies)} saved movies**")

        for batch_start in range(0, len(saved_movies), 5):
            batch = saved_movies.iloc[batch_start:batch_start + 5]
            columns = st.columns(5)

            for col, (_, movie) in zip(columns, batch.iterrows()):
                with col:
                    st.image(
                        movie_poster(movie, api_key),
                        width="stretch",
                    )

                    st.markdown(
                        f"**{escape(str(movie['clean_title']))}**"
                    )

                    st.caption(
                        f"{movie_year(movie)} · {movie_genres(movie)}"
                    )

                    if st.button(
                        "More info",
                        key=f"saved_detail_{int(movie['movieId'])}",
                        width="stretch",
                    ):
                        open_movie(movie["movieId"])
                        st.rerun()

                    if st.button(
                        "✕ Remove",
                        key=f"saved_remove_{int(movie['movieId'])}",
                        width="stretch",
                    ):
                        remove_from_watchlist(
                            username,
                            movie["movieId"],
                        )
                        st.rerun()


# =========================================================
# 17. MOVIE DETAILS PAGE
# =========================================================

elif st.session_state.navigation == "Movie details":

    selected_id = st.session_state.selected_movie_id

    if selected_id is None or selected_id not in movie_id_to_index:
        st.title("Movie Details")
        st.info(
            "Choose a movie from the Home or Discover page first."
        )

        if st.button("Return Home"):
            open_page("Home")
            st.rerun()

    else:
        if st.button("← Back to Home"):
            open_page("Home")
            st.rerun()

        selected_index = movie_id_to_index[selected_id]
        selected_movie = movies.iloc[selected_index]

        render_movie_details(selected_movie, username)


# =========================================================
# 18. HOW IT WORKS AND FOOTER
# =========================================================

with st.expander("🧠 How CineMatch AI works"):

    st.markdown(
        """
        **1. Movie features:** The system uses movie titles, genres,
        and user-generated tags.

        **2. TF-IDF:** Text features are converted into numerical vectors.

        **3. Cosine similarity:** The app compares the selected movie
        against other movies in the catalog.

        **4. Recommendations:** Similar titles are ranked and displayed.

        **5. MovieLens ratings:** Average ratings and rating counts
        come from the dataset.

        **6. My List and reviews:** Saved titles and written reviews
        are stored in a local SQLite database.

        This version uses content-based recommendations. It does not
        yet train a personalized model from your viewing history.
        """
    )

st.markdown(
    """
    <div class="footer">
        CINEMATCH · MOVIES MADE PERSONAL<br>
        Movie data: MovieLens / GroupLens Research.
        Posters: TMDB when configured.
        Not endorsed or certified by TMDB.
    </div>
    """,
    unsafe_allow_html=True,
)