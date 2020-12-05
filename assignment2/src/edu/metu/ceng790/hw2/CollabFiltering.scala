package edu.metu.ceng790.hw2

import edu.metu.ceng790.hw2.ParameterTuningALS._

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation._
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD

import java.io.PrintWriter
import java.nio.file.Files
import java.nio.file.Paths
import java.io.File

import scala.collection.mutable.ListBuffer
import scala.util.Random.shuffle
import scala.io.StdIn.readLine
import scala.reflect.io.Directory

// Part 1 - Collaborative Filtering
// Now that you have experimented with ALS, you know which parameters should be used to configure the algorithm. Create
// another file named CollabFiltering.scala. Our objective now is to go beyond aggregated performance measures such as
// MSE, and see if you would be satisfied by the recommendations of the system you just built. To do this, you need to
// add you as a user in the dataset.
object CollabFiltering {

  // Indices of Field Names in movies.csv
  val indexMovieId_movies: Int = 0
  val indexTitle_movies: Int = 1
  val indexGenres_movies: Int = 2

  // Indices of Field Names in userInputRatingsFile
  val indexMovieId_userInputRatings: Int = 0
  val indexTitle_userInputRatings: Int = 1
  val indexRating_userInputRatings: Int = 2

  val recommendationsDir: String = "recommendations/"
  val userInputRatingsFileName: String = "userInputRatings.csv"
  val modelParameters: Parameter = new Parameter(32, 60, 0.01)

  val countOfTopMovies: Int = 100
  val countOfTopMoviesToBeRatedByTheUser: Int = 25 // countOfTopMoviesToBeRatedByTheUser <= countOfTopMovies
  val countOfRecommendedMovies: Int = 20

  def main(args: Array[String]): Unit = {

    var spark: SparkSession = null
    var sc: SparkContext = null
    try {
      spark = SparkSession.builder()
        .appName("Getting your own recommendations")
        .config("spark.master", "local[*]")
        .getOrCreate()
      sc = spark.sparkContext
      sc.setCheckpointDir(modelsCheckpointDir)

      var rateAgain: String = ""
      // If User Ratings File and Related Matrix Factorization Model exists, ask the user whether he/she want to rate
      // again. If he/she chooses to rate again, get the new ratings and, train and save a new model. Otherwise, load
      // the existing model.
      if (Files.exists(Paths.get(recommendationsDir + userInputRatingsFileName))
        && Files.exists(Paths.get(recommendationsDir + modelParameters.getShortDescription))) {
        while (!rateAgain.equalsIgnoreCase("Y") && !rateAgain.equalsIgnoreCase("N")) {
          rateAgain = readLine("Do you want to rate movies again? (Y/N): ")
        }
      } else {
        rateAgain = "Y"
      }

      var model: MatrixFactorizationModel = null
      val modelPath = recommendationsDir + modelParameters.getShortDescription
      var userInputRatings: RDD[(Int, String, Int)] = null

      val originalMovies = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .option("header", "true")
        .load("ml-20m/movies.csv")
      println("Schema of movies.csv:")
      originalMovies.printSchema()

      // 2. Build the movies Map[Int,String] that associates a movie identifier to the movie title. This data is
      // available in movies.csv. Our goal is now to select which movies you will rate to build your user profile.
      // Since there are 27k movies in the dataset, if we select these movies at random, it is very likely that you
      // will not know about them. Instead, you will select the 100 most famous movies and rate 25 among them.
      val movies = originalMovies.rdd
        .map(r => (r.getInt(indexMovieId_movies), r.getString(indexTitle_movies)))
        .collectAsMap()

      // Get user ratings from the user and, train and save a new model for the user
      if (rateAgain.equalsIgnoreCase("Y")) {
        val originalRatings = spark.read
          .format("csv")
          .option("inferSchema", "true")
          .option("delimiter", ",")
          .option("header", "true")
          .load("ml-20m/ratings.csv")
        println("Schema of ratings.csv:")
        originalRatings.printSchema()

        val movieRatings = originalRatings.rdd
          .map(r => (r.getInt(indexMovieId_ratings), r.getDouble(indexRating_ratings)))

        // 3. Build mostRatedMovies that contains the 100 movies that were rated by the most users. This is very similar
        // to word-count, and finding the most frequent words in a document.
        val moviesWithTitleAndRatingCounts = movieRatings.groupByKey()
          .map{ case (movieId, mRatings) => (movieId, movies.get(movieId).get, mRatings.size) }
        val mostRatedMovies = moviesWithTitleAndRatingCounts
          .sortBy({ case (_, _, count) => count }, ascending = false)
          .map{ case (movieId, title, _) => (movieId, title)}
          .take(countOfTopMovies).toList

        // Obtain selectedMovies List[(Int, String)] that contains 25 movies selected at random in mostRatedMovies as
        // well as their title. To select elements at random in a list, a good strategy is to shuffle the list (i.e. put
        // it in a random order) and take the first elements. Shuffling the list can be done with
        // scala.util.Random.shuffle.
        val selectedMovies = shuffle(mostRatedMovies).take(countOfTopMoviesToBeRatedByTheUser)

        // 4. You can now use your recommender system by executing the program you wrote! Write a function
        // getRatings(selectedMovies) gives you 25 movies to rate and you can answer directly in the console in the
        // Scala IDE. Give a rating from 1 to 5, or 0 if you do not know this movie.
        val userRatedMovies = getRatings(selectedMovies)

        if (Files.notExists(Paths.get(recommendationsDir))) {
          Files.createDirectory(Paths.get(recommendationsDir))
        }
        val userInputRatingsFile = new PrintWriter(recommendationsDir + userInputRatingsFileName)
        userInputRatingsFile.write("movieId,title,rating\n")
        for ((movieId, title, rating) <- userRatedMovies.filter( { case (_, _, rating) => rating != 0 })) {
          if (title.contains(","))
            userInputRatingsFile.write("%d,\"%s\",%d\n".format(movieId, title, rating))
          else
            userInputRatingsFile.write("%d,%s,%d\n".format(movieId, title, rating))
        }
        userInputRatingsFile.close()

        userInputRatings = sc.parallelize(userRatedMovies)
          .filter{ case (_, _, rating) => rating != 0 } // Filter ratings for unknown movies
        val userRatingAverage = userInputRatings.map{ case (_, _, rating) => rating }.mean()
        val userRatings = userInputRatings.map{ case (movieId, _, rating) => Rating(0, movieId, rating/userRatingAverage) }

        // Normalize the ratings per user with avgRatingPerUser
        val avgRatingPerUser = originalRatings.rdd
          .map(r => Rating(r.getInt(indexUserId_ratings), r.getInt(indexMovieId_ratings), r.getDouble(indexRating_ratings)))
          .map(r => (r.user, r.rating)).groupByKey().map(r => (r._1, r._2.sum/r._2.size))
        val ratings = originalRatings.rdd
          .map(r => Rating(r.getInt(indexUserId_ratings), r.getInt(indexMovieId_ratings), r.getDouble(indexRating_ratings)))
          .map(r => (r.user, Rating(r.user, r.product, r.rating))).join(avgRatingPerUser)
          .map{ case (_, (Rating(u, p, r), userAvg)) => Rating(u, p, r/userAvg) }
        val trainingRatings = ratings.union(userRatings)

        println("ALS Model with %s is training.".format(modelParameters.toString))
        model = ALS.train(trainingRatings, modelParameters.rank, modelParameters.iteration, modelParameters.lambda)
        if (Files.exists(Paths.get(modelPath))) {
          val directory = new Directory(new File(modelPath))
          directory.deleteRecursively()
        }
        model.save(sc, modelPath)
        println("ALS Model with %s is saved to \"%s\".".format(modelParameters.toString, modelPath))
      } else {  // User doesn't want to rate the movies again, load the user ratings and the trained model for the user
        val inputRatings = spark.read
          .format("csv")
          .option("inferSchema", "true")
          .option("delimiter", ",")
          .option("header", "true")
          .load(recommendationsDir + userInputRatingsFileName)
        println("Schema of %s:".format(userInputRatingsFileName))
        inputRatings.printSchema()

        userInputRatings = inputRatings.rdd
          .map(r => (r.getInt(indexMovieId_userInputRatings), r.getString(indexTitle_userInputRatings), r.getInt(indexRating_userInputRatings)))

        model = MatrixFactorizationModel.load(sc, modelPath)
        println("ALS Model with %s is loaded from \"%s\".".format(modelParameters.toString, modelPath))
      }

      // 5. After finishing the rating, your program should display the top 20 movies that you might like. Look at the
      // recommendations, are you happy about your recommendations? Comment.
      val userInputMoviesWithTitles = userInputRatings.map{ case (movieId, title, _) => (movieId, title)}
      val countOfUserInputs = userInputMoviesWithTitles.count().toInt
      val recommendations = sc.parallelize(model.recommendProducts(0, countOfRecommendedMovies + countOfUserInputs))
        .map(r => (r.product, movies.get(r.product).get))
        .subtract(userInputMoviesWithTitles)  // Subtract the already rated movies by the user from the recommendation list

      var index: Int = 1
      println("Top %d Recommendations:".format(countOfRecommendedMovies))
      recommendations.take(countOfRecommendedMovies).foreach{
        case (_, title) => println("%d. %s".format(index, title))
        index = index + 1
      }
    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }

  // getRatings(selectedMovies) function gives you 25 movies to rate and you can answer directly in the console in the
  // Scala IDE. Give a rating from 1 to 5, or 0 if you do not know this movie.
  def getRatings(selectedMovies: List[(Int, String)]): List[(Int, String, Int)] = {
    var userRatings = new ListBuffer[(Int, String, Int)]()
    println("Please, rate the following movies in the scale of 1 to 5. " +
      "If you do not know the movie, rate 0 or just press enter.")
    for ((movieId, title) <- selectedMovies) {
      var rating = 0
      var isValid = false

      while (!isValid) {
        val input = readLine("%s: ".format(title))
        try {
          if (input.equals("")) {
            rating = 0
            isValid = true
          } else {
            rating = input.toInt
            if (rating >= 0 && rating <= 5)
              isValid = true
            else
              throw new java.lang.Exception()
          }
        } catch {
          case e: Exception => println("Invalid input rating \"%s\", please rate again.".format(input))
        }
      }
      userRatings += Tuple3(movieId, title, rating)
    }
    userRatings.toList
  }
}
