package edu.metu.ceng790.hw2

import edu.metu.ceng790.hw2.ParameterTuningALS._
import edu.metu.ceng790.hw2.CollabFiltering._

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

// Part 2 - Content-based Nearest Neighbors
// For collaborative filtering, you relied purely on rankings and did not use movie attributes (genres) at all. For this
// part of the assignment, you will use a different method: content-based recommendation. Our goal here is to build for
// each user a vector of features (genres) describing their interests. Then, we will find the k users that are most
// similar using those vectors and cosine similarity to obtain a recommendation.
object NearestNeighbors {

  val k_NearestNeighbors: Int = 20

  def main(args: Array[String]): Unit = {

    var spark: SparkSession = null
    var sc: SparkContext = null
    try {
      spark = SparkSession.builder()
        .appName("Content-based Nearest Neighbors")
        .config("spark.master", "local[*]")
        .getOrCreate()
      sc = spark.sparkContext

      val originalRatings = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .option("header", "true")
        .load("ml-20m/ratings.csv")
      println("Schema of ratings.csv:")
      originalRatings.printSchema()

      val originalMovies = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .option("header", "true")
        .load("ml-20m/movies.csv")
      println("Schema of movies.csv:")
      originalMovies.printSchema()

      val inputRatings = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .option("header", "true")
        .load(recommendationsDir + userInputRatingsFileName)
      println("Schema of %s:".format(userInputRatingsFileName))
      inputRatings.printSchema()

      // 1. For this part, you will transform ratings into binary information. There are movies the user liked and
      // movies the user did not like. In a file named NearestNeighbors.scala, build the goodRatings RDD by transforming
      // the ratings RDD to only keep, for each user, ratings that are above their average. For instance, if a user
      // rates on average 2.8, we only keep their ratings that are greater or equal to 2.8.
      val ratings = originalRatings.rdd
        .map(r => Rating(r.getInt(indexUserId_ratings), r.getInt(indexMovieId_ratings), r.getDouble(indexRating_ratings)))
      val userAverageRatings = ratings
        .map(r => (r.user, r.rating))
        .groupByKey()
        .map{ case (userId, userRatings) => (userId, userRatings.sum/userRatings.size) }
      val goodRatings = ratings
        .map(r => (r.user, (r.product, r.rating)))
        .join(userAverageRatings)
        .filter{ case (_, ((_, rating), userAvgRating)) => rating >= userAvgRating }
        .map{ case (userId, ((movieId, rating), _)) => (userId, movieId, rating) }

      // 2. Build the movieNames Map[Int,String] that associates a movie identifier to the movie name. You have already
      // done this in the previous part of this assignment.
      val movies = originalMovies.rdd
        .map(r => (r.getInt(indexMovieId_movies), r.getString(indexTitle_movies), r.getString(indexGenres_movies)))
      val movieNames = movies.map{ case (movieId, title, _) => (movieId, title) }
        .collectAsMap()

      // 3. Build the movieGenres Map[Int, Array[String]] that associates a movie identifier to the list of genres it
      // belongs to. This information is available in the movies.csv file, in the third column, and movies are separated
      // by "|". If you use split, you will need to write "\\|" as a parameter.
      val movieGenres = movies.map{ case (movieId, _, genres) => (movieId, genres.split("\\|")) }
        .collectAsMap()

      // 4. Provide the code that builds the userVectors RDD. This RDD contains (Int, Map[String, Int]) pairs in which
      // the first element is a user ID, and the second element is the vector describing the user. If a user has liked 2
      // action movies, then this vector will contain an entry (“action”, 2). Write the userSim function that computes
      // the cosine similarity between two user vectors. The mathematical formula is available on the slides. To perform
      // a square root operation, use Math.sqrt(x).
      val userVectors = goodRatings.map{ case (userId, movieId, _) => (userId, movieGenres.get(movieId).get) }
        .flatMapValues(r => r)
        .groupBy(r => r)
        .map{ case ((userId, genre), groupedValues) => (userId, (genre, groupedValues.size)) }
        .groupByKey()
        .mapValues(r => r.toMap)

      // 5. Now, write a function named knn that takes a user profile named testUser. Then the function selects the list
      // of k users that are most similar to the testUser, and returns recommendation, the list of movies recommended to
      // the user.
      val userInputRatings = inputRatings.rdd
        .map(r => (r.getInt(indexMovieId_userInputRatings), r.getString(indexTitle_userInputRatings), r.getInt(indexRating_userInputRatings)))
      val userInputMovies = userInputRatings.map(r => r._1).collect()

      val userInputAverageRating = userInputRatings.map(r => (r._3)).mean()
      val userInputGoodRatings = userInputRatings.filter(r => r._3 >= userInputAverageRating)
      val userVector = (0, userInputGoodRatings.map{ case (movieId, _, _) => (movieGenres.get(movieId).get) }
        .flatMap(r => r)
        .map(r => (r, 1))
        .reduceByKey(_ + _)
        .collect().toMap)

      val similarUsers = knn(userVector, userVectors)
      //val goodRatingsOfSimilarUsers = goodRatings.filter{ case (userId, _, _) => similarUsers.contains(userId) }
      val ratingsOfSimilarUsers = ratings.map(r => (r.user, r.product, r.rating) )  // Instead of good ratings! TODO
        .filter{ case (userId, _, _) => similarUsers.contains(userId) }
        .filter{ case (_, movieId, _) => !userInputMovies.contains(movieId) }
        .map{ case (_, movieId, rating) => (movieId, rating) }
        .groupByKey()
        .map{ case (movieId, movieRatings) => (movieId, movieNames.get(movieId).get, movieRatings.sum/movieRatings.size) }
        .sortBy({ case (_, _, avgRating) => avgRating }, ascending = false)

      var index: Int = 1
      println("Top 20 Recommendations:")
      ratingsOfSimilarUsers.take(20).foreach {
        case (_, title, _) => println("%d. %s".format(index, title))
          index = index + 1
      }
    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }

  // userSim function that computes the cosine similarity between two user vectors. The mathematical formula is
  // available on the slides. To perform a square root operation, use Math.sqrt(x).
  def userSim(userVector1: Map[String, Int], userVector2: Map[String, Int]): Double = {
    var dotProduct: Int = 0
    var sumSquares1: Double = 0.0
    var sumSquares2: Double = 0.0
    var vector1: Map[String, Int] = null
    var vector2: Map[String, Int] = null

    if (userVector1.size < userVector2.size) {
      vector1 = userVector1
      vector2 = userVector2
    } else {
      vector1 = userVector2
      vector2 = userVector1
    }

    for (genre <- vector1.keys) {
      val val1: Int = vector1.get(genre).get
      sumSquares1 = sumSquares1 + Math.pow(val1, 2)

      if (vector2.contains(genre)) {
        val val2: Int = vector2.get(genre).get
        dotProduct = dotProduct + val1 * val2
      }
    }

    for (genre <- vector2.keys) {
      val val2: Int = vector2.get(genre).get
      sumSquares2 = sumSquares2 + Math.pow(val2, 2)
    }

    dotProduct / (Math.sqrt(sumSquares1) * Math.sqrt(sumSquares2))
  }

  // knn function that takes a user profile named testUser. Then, the function selects the list of k
  // users that are most similar to the testUser, and returns recommendation, the list of movies recommended to the user.
  def knn(testUser: (Int, Map[String, Int]), userVectors: RDD[(Int, Map[String, Int])]): Array[Int] = {
    val testUserId = testUser._1
    val testUserVector = testUser._2

    val similarities = userVectors.filter{ case (userId, _) => userId != testUserId }
      .map{ case (userId, userVector) => (userId, userVector, userSim(testUserVector, userVector)) }
      .sortBy(r => r._3, ascending = false)

    similarities.map(r => r._1).take(k_NearestNeighbors)
  }
}
