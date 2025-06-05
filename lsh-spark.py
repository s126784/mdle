import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MovieLens CF with LSH Parameters")

    parser.add_argument("--bucketLength", type=float, default=2.0, help="Bucket length for LSH")
    parser.add_argument("--numHashTables", type=int, default=3, help="Number of hash tables for LSH")
    parser.add_argument("--threshold", type=float, default=1.0, help="Similarity threshold for approximate join")
    parser.add_argument("--topK", type=int, default=20, help="Top K similar items to consider")
    parser.add_argument("--defaultRating", type=float, default=3.0, help="Default rating if no prediction available")
    parser.add_argument("--dataset", type=str, default="100k", help="Dataset subfolder name (e.g. 100k, 1M)")
    parser.add_argument("--outputDir", type=str, default="results", help="Directory for output files")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.environ["PYSPARK_PYTHON"] = "/opt/miniconda3/envs/3_9/bin/python"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/miniconda3/envs/3_9/bin/python"

    import time
    import pandas as pd
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import collect_set, avg, row_number, struct, collect_list, col, count as sql_count, sum as sql_sum, coalesce, lit
    from pyspark.sql.window import Window
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    from itertools import combinations
    import matplotlib.pyplot as plt
    import numpy as np
    from pyspark.ml.feature import Normalizer, BucketedRandomProjectionLSH
    from pyspark.ml.evaluation import RegressionEvaluator
    import argparse

    parser = argparse.ArgumentParser(description="MovieLens CF with LSH Parameters")

    parser.add_argument("--bucketLength", type=float, default=2.0, help="Bucket length for LSH")
    parser.add_argument("--numHashTables", type=int, default=3, help="Number of hash tables for LSH")
    parser.add_argument("--threshold", type=float, default=1.0, help="Similarity threshold for approximate join")
    parser.add_argument("--topK", type=int, default=20, help="Top K similar items to consider")
    parser.add_argument("--defaultRating", type=float, default=3.0, help="Default rating if no prediction available")
    parser.add_argument("--dataset", type=str, default="100k", help="Dataset subfolder name (e.g. 100k, 1M)")
    parser.add_argument("--outputDir", type=str, default="results", help="Directory for output files")

    args = parser.parse_args()

    seed = 126784

    BUCKET_LENGTH = args.bucketLength
    NUM_HASH_TABLES = args.numHashTables
    SIMILARITY_THRESHOLD = args.threshold
    TOP_K = args.topK
    DEFAULT_RATING = args.defaultRating
    DATASET = args.dataset
    OUTPUT_DIR = args.outputDir
    METRICS_FILE = os.path.join(OUTPUT_DIR, "100k_metrics.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # %%
    spark = SparkSession.builder \
        .appName("126784-PRJ3B") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()
    sc = spark.sparkContext

    # %%
    ratings_df = spark.read.csv(f"./datasets/{DATASET}/ratings.csv", header=True, inferSchema=True)
    movies_df = spark.read.csv(f"./datasets/{DATASET}/movies.csv", header=True, inferSchema=True)

    print(f"Ratings count: {ratings_df.count()}")
    print(f"Movies count: {movies_df.count()}")
    print(f"Users count: {ratings_df.select('userId').distinct().count()}")

    rating_stats_pd = ratings_df.groupBy("rating").count().orderBy("rating").toPandas()
    plt.figure(figsize=(5, 1))
    plt.bar(rating_stats_pd['rating'], rating_stats_pd['count'])
    plt.title('Rating Distribution')


    # %% [markdown]
    # ### Sample Rating

    # %%
    ratings_df.show(5)

    # %% [markdown]
    # ### Sample Movies

    # %%
    movies_df.show(5, truncate=False)

    # %%
    ratings_train_df, ratings_test_df = ratings_df.randomSplit([0.9, 0.1], seed=seed)

    print("Training set statistics:")
    print(f"Total ratings: {ratings_train_df.count()}")
    print(f"Unique users: {ratings_train_df.select('userId').distinct().count()}")
    print(f"Unique movies: {ratings_train_df.select('movieId').distinct().count()}")
    print(f"Average rating: {ratings_train_df.agg(avg('rating')).collect()[0][0]:.3f}")

    print("\nTest set statistics:")
    print(f"Total ratings: {ratings_test_df.count()}")
    print(f"Unique users: {ratings_test_df.select('userId').distinct().count()}")
    print(f"Unique movies: {ratings_test_df.select('movieId').distinct().count()}")
    print(f"Average rating: {ratings_test_df.agg(avg('rating')).collect()[0][0]:.3f}")

    # %%
    user_index = ratings_train_df.select("userId").distinct().withColumn(
        "user_idx",
        row_number().over(Window.orderBy("userId")) - 1
    )
    num_users = user_index.count()
    user_index.show(10)

    # %%
    train_with_idx = ratings_train_df.join(user_index, on="userId")
    train_with_idx.toPandas().sample(5)

    # %%
    item_user_ratings = train_with_idx.groupBy("movieId") \
        .agg(
            collect_list(struct("user_idx", "rating")).alias("user_ratings"),
            sql_count("*").alias("num_ratings")
        )
    item_user_ratings.select("movieId", "num_ratings") \
        .orderBy(col("num_ratings").desc()).show(10)

    # %%
    MIN_RATINGS = 5
    filtered_items = item_user_ratings.filter(col("num_ratings") >= MIN_RATINGS)
    print(f"Movies with more than {MIN_RATINGS} ratings: {filtered_items.count()}")

    # %%
    def to_sparse_vector(user_ratings, size):
        """
        Converts a list of user ratings into a sparse vector.
        user_ratings: list of structs {user_idx, rating}
        size: total number of users
        """
        if not user_ratings:
            return Vectors.sparse(size, [], [])

        # sorting to make all one style
        sorted_ratings = sorted(user_ratings, key=lambda x: x.user_idx)

        indices = [r.user_idx for r in sorted_ratings]
        values = [float(r.rating) for r in sorted_ratings]

        return Vectors.sparse(size, indices, values)

    # %%

    def convert_row_to_vector(row, size):
        return Row(
            movieId=row["movieId"],
            features=to_sparse_vector(row["user_ratings"], size),
            num_ratings=row["num_ratings"]
        )

    # item_vectors_rdd = filtered_items.rdd.map(
    #     lambda row: Row(
    #         movieId=row["movieId"],
    #         features=to_sparse_vector(row["user_ratings"], num_users),
    #         num_ratings=row["num_ratings"]
    #     )
    # )
    item_vectors_rdd = filtered_items.rdd.map(lambda row: convert_row_to_vector(row, num_users))


    item_vectors_df = spark.createDataFrame(item_vectors_rdd)

    print(f"{item_vectors_df.count()} vectors created")
    item_vectors_df.select("movieId", "num_ratings", "features").show(10)

    # %%
    # data should be normalized for LSH
    normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
    normalized_df = normalizer.transform(item_vectors_df)
    normalized_df.select("movieId", "num_ratings", "norm_features").show(10)

    # %%
    lsh = BucketedRandomProjectionLSH(
        inputCol="norm_features", outputCol="hashes", bucketLength=BUCKET_LENGTH, numHashTables=NUM_HASH_TABLES, seed=seed
    )
    lsh_model = lsh.fit(normalized_df)
    hashed_df = lsh_model.transform(normalized_df)

    similar_pairs = lsh_model.approxSimilarityJoin(
        normalized_df, normalized_df, SIMILARITY_THRESHOLD, distCol="distance"
    ).filter(col("datasetA.movieId") < col("datasetB.movieId"))

    similarities_df = similar_pairs.withColumn(
        "cosine_similarity", 1 - (col("distance") ** 2) / 2
    ).select(
        col("datasetA.movieId").alias("movie_i"),
        col("datasetB.movieId").alias("movie_j"),
        "cosine_similarity"
    )

    # %%

    reverse_similarities = similarities_df.selectExpr(
        "movie_j as movie_i", "movie_i as movie_j", "cosine_similarity"
    )
    full_similarities = similarities_df.union(reverse_similarities)

    # get top-20 similar movies for each movie
    window_spec = Window.partitionBy("movie_i").orderBy(col("cosine_similarity").desc())
    final_similarities = full_similarities.withColumn("rank", row_number().over(window_spec)) \
                                        .filter(col("rank") <= TOP_K) \
                                        .drop("rank").cache()


    # %%

    def find_similar_movies(movie_id, k=5):
        """
        Find top-k similar movies for a given movie_id.
        """
        return final_similarities.filter(col("movie_i") == movie_id) \
            .join(movies_df, col("movie_j") == movies_df.movieId) \
            .orderBy(col("cosine_similarity").desc()) \
            .select("movie_j", "title", "cosine_similarity") \
            .limit(k).collect()


    # %%

    test_movie = ratings_train_df.groupBy("movieId").count().orderBy(col("count").desc()).first().movieId
    test_movie_title = movies_df.filter(col("movieId") == test_movie).select("title").first().title


    print(f"Test movie: {test_movie_title} (ID: {test_movie})")
    for i, row in enumerate(find_similar_movies(test_movie), 1):
        print(f"{i}. {row.title} (similarity: {row.cosine_similarity:.3f})")

    # %%
    test_with_idx = ratings_test_df.join(user_index, on="userId", how="inner")


    # %%
    # join test with similarities to find neighbor movies for each test rating
    test_with_similarities = test_with_idx.alias("test") \
        .join(final_similarities.alias("sim"),
            col("test.movieId") == col("sim.movie_i")) \
        .select(
            col("test.userId"),
            col("test.movieId").alias("target_movie"),
            col("test.rating").alias("true_rating"),
            col("sim.movie_j").alias("neighbor_movie"),
            col("sim.cosine_similarity")
        )

    # %%
    # join with training ratings to get user's ratings for neighbor movies
    test_with_neighbor_ratings = test_with_similarities.alias("tsim") \
        .join(train_with_idx.alias("train"),
            (col("tsim.userId") == col("train.userId")) &
            (col("tsim.neighbor_movie") == col("train.movieId"))) \
        .select(
            col("tsim.userId"),
            col("tsim.target_movie"),
            col("tsim.true_rating"),
            col("tsim.neighbor_movie"),
            col("tsim.cosine_similarity"),
            col("train.rating").alias("neighbor_rating")
        )

    # %%
    # calculate weighted average predictions
    predictions_df = test_with_neighbor_ratings.groupBy("userId", "target_movie", "true_rating") \
        .agg(
            (sql_sum(col("cosine_similarity") * col("neighbor_rating")) /
            sql_sum(col("cosine_similarity"))).alias("predicted_rating")
        )

    # %%
    final_predictions = test_with_idx.alias("test") \
        .join(predictions_df.alias("pred"),
            (col("test.userId") == col("pred.userId")) &
            (col("test.movieId") == col("pred.target_movie")),
            how="left") \
        .select(
            col("test.userId"),
            col("test.movieId"),
            col("test.rating").alias("true_rating"),
            coalesce(col("pred.predicted_rating"), lit(DEFAULT_RATING)).alias("predicted_rating")
        )

    print("Generated predictions for", final_predictions.count(),  "test ratings")
    final_predictions.show(10)

    # %%
    final_predictions.toPandas().to_csv(os.path.join(OUTPUT_DIR, "100k_predictions.csv"), index=False)

    # %%
    rmse_evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="true_rating", predictionCol="predicted_rating"
    )
    mae_evaluator = RegressionEvaluator(
        metricName="mae", labelCol="true_rating", predictionCol="predicted_rating"
    )

    rmse = rmse_evaluator.evaluate(final_predictions)
    mae = mae_evaluator.evaluate(final_predictions)

    print(f"Final Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # %%
    global_avg = ratings_train_df.agg(avg("rating")).collect()[0][0]
    neighbor_predictions = final_predictions.filter(col("predicted_rating") != DEFAULT_RATING).count()
    total_predictions = final_predictions.count()
    coverage = neighbor_predictions / total_predictions * 100

    print(f"Coverage: {coverage:.1f}% ({neighbor_predictions}/{total_predictions} predictions from neighbors)")


    # %% [markdown]
    #



    # %%
    results_row = pd.DataFrame([{
        "dataset": "100k",
        "rmse": rmse,
        "mae": mae,
        "bucketLength": BUCKET_LENGTH,
        "numHashTables": NUM_HASH_TABLES,
        "threshold": SIMILARITY_THRESHOLD,
        "top_k": TOP_K,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }])
    if os.path.exists(METRICS_FILE):
        old = pd.read_csv(METRICS_FILE)
        all_metrics = pd.concat([old, results_row], ignore_index=True)
    else:
        all_metrics = results_row

    all_metrics.to_csv(METRICS_FILE, index=False)

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    # spark.stop()



