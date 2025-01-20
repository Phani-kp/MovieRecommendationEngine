from pyspark.sql import SparkSession

def recommend_movies(model, user_id, movies_df, n=5):
    # Recommend movies for a given user
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recommendations = model.recommendForUserSubset(user_df, n).collect()

    for rec in recommendations:
        movie_ids = [row.movieId for row in rec.recommendations]
        print("Recommended Movies:")
        movies_df.filter(col("movieId").isin(movie_ids)).show()

if __name__ == "__main__":
    spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()
    model_path = "model/als_model"
    movies_path = "data/movies.csv"

    # Load model and movies data
    model = ALS.load(model_path)
    movies_df = spark.read.csv(movies_path, header=True, inferSchema=True)

    user_id = 1  # Change as needed
    recommend_movies(model, user_id, movies_df)

    spark.stop()
