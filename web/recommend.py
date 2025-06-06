import pandas as pd

ratings = pd.read_csv("./ds/ratings.csv")
movies = pd.read_csv("./ds/movies.csv")

# Load precomputed LSH recommendations
recs_df = pd.read_parquet("./cache/user_recommendations.parquet")
movie_sims = pd.read_parquet("./cache/movie_similarities.parquet")


def get_user_ids():
    return sorted(ratings["userId"].unique())

def get_user_ratings(user_id):
    user_ratings_df = (
        ratings[ratings.userId == user_id]
        .merge(movies, on="movieId")[["title", "rating"]]
        .sort_values("rating", ascending=False)
    )

    # Convert to list of dictionaries and add average rating
    user_ratings_list = user_ratings_df.to_dict('records')

    # Calculate average rating
    if user_ratings_list:
        avg_rating = user_ratings_df['rating'].mean()
        total_ratings = len(user_ratings_list)
    else:
        avg_rating = 0
        total_ratings = 0

    # Return both the list and stats
    return {
        'ratings': user_ratings_list,
        'avg_rating': avg_rating,
        'total_ratings': total_ratings
    }

def get_recommendations(user_id, top_n=5):
    return (
        recs_df[recs_df.userId == user_id]
        .sort_values("score", ascending=False)
        .head(top_n)["title"]
        .tolist()
    )


def get_all_movie_titles():
    return (
        movie_sims[["movie_i", "title_i"]]
        .drop_duplicates()
        .sort_values("title_i")
        .rename(columns={"movie_i": "movieId", "title_i": "title"})
        .to_dict("records")
    )

def recommend_based_on_movies(selected_titles, top_n=5):
    if not selected_titles:
        return []

    # Filter relevant similarities
    sim_matches = movie_sims[movie_sims["title_i"].isin(selected_titles)]

    # Aggregate and score recommended movies
    recs = (
        sim_matches.groupby(["movie_j", "title_j"])
        .agg({"cosine_similarity": "sum"})
        .reset_index()
        .rename(columns={"cosine_similarity": "score"})
    )

    # Exclude already selected
    recs = recs[~recs["title_j"].isin(selected_titles)]

    # Return top N
    return recs.sort_values("score", ascending=False).head(top_n).to_dict("records")
