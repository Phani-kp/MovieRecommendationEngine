from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def preprocess_data(spark, movies_path, ratings_path):
    # Load data
    movies_df = spark.read.csv(movies_path, header=True, inferSchema=True)
    ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

    # Drop unnecessary columns
    ratings_df = ratings_df.drop("timestamp")

    # Print schema
    print("Movies Schema:")
    movies_df.printSchema()

    print("Ratings Schema:")
    ratings_df.printSchema()

    return movies_df, ratings_df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("MovieRecommendationPreprocess").getOrCreate()
    movies_path = "data/movies.csv"
    ratings_path = "data/ratings.csv"

    preprocess_data(spark, movies_path, ratings_path)
    spark.stop()
