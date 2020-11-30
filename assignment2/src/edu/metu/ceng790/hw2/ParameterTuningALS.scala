package edu.metu.ceng790.hw2

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.rdd.RDD._
import java.nio.file.Paths
import java.nio.file.Files
import java.io.PrintWriter
import scala.math.pow

// Part 1 - Collaborative Filtering
object ParameterTuningALS {

  // Indices of Field Names in ratings.csv
  val indexUserId: Int = 0
  val indexMovieId: Int = 1
  val indexRating: Int = 2
  val indexTimestamp: Int = 3

  val modelsOutputDir: String = "models/"
  val modelsCheckpointDir: String = "checkpoints/"
  val reportsOutputDir: String = "reports/"
  val reportFileName: String = "mseScores_%s.csv" //"mseScores_%s_custom.csv"

  def main(args: Array[String]): Unit = {

    var spark: SparkSession = null
    var sc: SparkContext = null
    try {
      spark = SparkSession.builder()
        .appName("Train a model and tune parameters")
        .config("spark.master", "local[*]")
        .getOrCreate()
      sc = spark.sparkContext
      sc.setCheckpointDir(modelsCheckpointDir)

      // Inferring the schema can be commented out as it requires reading the data one more time. In that case String to
      // Int and Double conversions should be made for the userId, movieId and rating fields' values.
      val originalRatings = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .option("header", "true")
        .load("ml-20m/ratings.csv")
      originalRatings.printSchema()

      // First, you need to load the dataset into the RDD Ratings. In order to get more accurate predictions, you should
      // normalize the ratings. Rating normalizations types:
      //   0) no normalization
      //   1) normalize the ratings per user with avgRatingPerUser,
      //   2) normalize the ratings to [0,1] by dividing the ratings with 5.

      // No normalization (Normalization Type 0)
      //val ratings = originalRatings.rdd
      //  .map(r => Rating(r.getInt(indexUserId), r.getInt(indexMovieId), r.getDouble(indexRating)))
      //val normalization : String = "Normalization0"

      // Normalize the ratings per user with avgRatingPerUser. (Normalization Type 1)
      val avgRatingPerUser = originalRatings.rdd
        .map(r => Rating(r.getInt(indexUserId), r.getInt(indexMovieId), r.getDouble(indexRating)))
        .map(r => (r.user, r.rating)).groupByKey().map(r => (r._1, r._2.sum/r._2.size))
      val ratings = originalRatings.rdd
        .map(r => Rating(r.getInt(indexUserId), r.getInt(indexMovieId), r.getDouble(indexRating)))
        .map(r => (r.user, Rating(r.user, r.product, r.rating))).join(avgRatingPerUser)
        .map{case (_, (Rating(u, p, r), userAvg)) => Rating(u, p, r/userAvg)}
      val normalization : String = "Normalization1"

      // Normalize the ratings to [0,1] by dividing the ratings with 5. (Normalization Type 2)
      //val ratings = originalRatings.rdd
      //  .map(r => Rating(r.getInt(indexUserId), r.getInt(indexMovieId), r.getDouble(indexRating)))
      //  .map(r => Rating(r.user, r.product, r.rating/5))
      //val normalization : String = "Normalization2"

      // Split the data in train and test sets of sizes 8:2. The 80% of the data will be used for training and tuning
      // the parameters, the 20% of the data will be used for testing.
      // For the consistency among splits at each run-time, seed value is set to 5.
      val Array(trainingRatings, testRatings) = ratings.randomSplit(Array(0.8, 0.2), seed = 5)

      // Parameters to decide which combination works well (i.e. lower MSE score with test data)
      val ranks: Array[Int] = Array(8, 12)
      val iterations: Array[Int] = Array(20, 30)
      val lambdas: Array[Double] = Array(0.01, 1.0, 10.0)

      if (Files.notExists(Paths.get(reportsOutputDir))) {
        Files.createDirectory(Paths.get(reportsOutputDir))
      }
      val reportFile = new PrintWriter(reportsOutputDir + reportFileName.format(normalization))
      reportFile.write("rank iteration lambda mse\n")

      // 1. Change the values for rank, lambda and iteration and create the cross product of 2 different ranks (8 and
      // 12), 3 different lambdas (0.01, 1.0 and 10.0), and two different numbers of iterations (20 and 30). What are
      // the values for the best model? Store these values, you will need them for the next question.
      for (r <- ranks; i <- iterations; l <- lambdas) {
      // For training with custom parameters
      //for ((r, i, l) <- Array((16,30,0.01), (12,40,0.01), (12,30,0.1), (16,40,0.01), (24,40,0.01), (32,60,0.01))) {
        val p = new Parameter(r, i, l)
        val modelPath = modelsOutputDir + p.getShortDescription + normalization

        var model: MatrixFactorizationModel = null
        // If the ALS Model is not exist, train one with given parameters and save.
        if (Files.notExists(Paths.get(modelPath))) {
          println("ALS Model with %s is training.".format(p.toString))
          model = ALS.train(trainingRatings, p.rank, p.iteration, p.lambda)
          model.save(sc, modelPath)
          println("ALS Model with %s is saved to \"%s\".".format(p.toString, modelPath))
        } else {  // Else, load the ALS Model with given parameters.
          model = MatrixFactorizationModel.load(sc, modelPath)
          println("ALS Model with %s is loaded from \"%s\".".format(p.toString, modelPath))
        }

        // Calculate Mean Square Error (MSE)
        val testUsersAndMovies = testRatings.map(r => (r.user, r.product))
        val validationTestRatings = testRatings.map(r => ((r.user, r.product), r.rating))
        val predictionTestRatings = model.predict(testUsersAndMovies).map(r => ((r.user, r.product), r.rating))
        val validationAndPredictionTestRatings = validationTestRatings.join(predictionTestRatings)
        val meanSquareError = validationAndPredictionTestRatings
          .map{case (_, (rating, prediction)) => pow(rating - prediction, 2)}.mean()

        println("MSE for ALS Model with %s is %f.\n".format(p.toString, meanSquareError))
        reportFile.write("%d %d %.2f %f\n".format(p.rank, p.iteration, p.lambda, meanSquareError))
      }

      reportFile.close()
    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }
}