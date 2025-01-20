from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def train_model(spark, ratings_path):
    # Load ratings data
    ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

    # Split data into training and testing
    training_df, test_df = ratings_df.randomSplit([0.8, 0.2])

    # Build ALS model
    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )

    # Train the model
    model = als.fit(training_df)

    # Make predictions
    predictions = model.transform(test_df)

    # Evaluate the model
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return model

if __name__ == "__main__":
    spark = SparkSession.builder.appName("MovieRecommendationTrain").getOrCreate()
    ratings_path = "data/ratings.csv"

    train_model(spark, ratings_path)
    spark.stop()
