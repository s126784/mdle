from flask import Flask, render_template, request, jsonify
from recommend import (
    get_user_ids, get_user_ratings, get_recommendations,
    get_all_movie_titles, recommend_based_on_movies
)

app = Flask(__name__)

@app.route("/")
def home():
    """Landing page with navigation options"""
    return render_template("home.html")

@app.route("/item-recommendations", methods=["GET", "POST"])
def item_recommendations():
    """Recommendations page"""
    user_ids = get_user_ids()
    selected_user = request.form.get("user_id", type=int, default=user_ids[0])

    user_data = get_user_ratings(selected_user)
    recommendations = get_recommendations(selected_user)

    return render_template(
        "item_recommendations.html",
        user_ids=user_ids,
        selected_user=selected_user,
        user_ratings=user_data['ratings'],
        avg_rating=user_data['avg_rating'],
        total_ratings=user_data['total_ratings'],
        recommendations=recommendations
    )

@app.route("/movie-based", methods=["GET", "POST"])
def movie_based():
    """Movie-based recommendations page"""
    all_movies = get_all_movie_titles()
    selected = request.form.getlist("selected_movies")
    recommendations = []

    if selected:
        recommendations = recommend_based_on_movies(selected, top_n=8)

    return render_template(
        "movie_based.html",
        all_movies=all_movies,
        selected_movies=selected,
        movie_recommendations=recommendations
    )

@app.route("/api/search-movies")
def search_movies():
    """API endpoint for movie search"""
    query = request.args.get('q', '').lower()
    all_movies = get_all_movie_titles()

    if query:
        filtered_movies = [
            movie for movie in all_movies
            if query in movie['title'].lower()
        ]
        return jsonify(filtered_movies[:20])  # Limit results
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=False)
