package edu.metu.ceng790.hw2

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer
import scala.util.Random.shuffle
import scala.io.StdIn.readLine

// Part 1 - Collaborative Filtering
object CollabFiltering {

  // Indices of Field Names in movies.csv
  val indexMovieId_movies: Int = 0
  val indexTitle_movies: Int = 1
  val indexGenres_movies: Int = 2

  def main(args: Array[String]): Unit = {

    var spark: SparkSession = null
    var sc: SparkContext = null
    try {
      spark = SparkSession.builder()
        .appName("Getting your own recommendations")
        .config("spark.master", "local[*]")
        .getOrCreate()
      sc = spark.sparkContext

      val originalMovies = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .option("header", "true")
        .load("ml-20m/movies.csv")
      originalMovies.printSchema()

      // 2. Build the movies Map[Int,String] that associates a movie identifier to the movie title. This data is
      // available in movies.csv. Our goal is now to select which movies you will rate to build your user profile. Since
      // there are 27k movies in the dataset, if we select these movies at random, it is very likely that you will not
      // know about them. Instead, you will select the 100 most famous movies and rate 25 among them.
      val movies = originalMovies.rdd
        .map(r => (r.getInt(indexMovieId_movies), r.getString(indexTitle_movies)))
        .collectAsMap()

      val originalRatings = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .option("header", "true")
        .load("ml-20m/ratings.csv")
      originalRatings.printSchema()

      val ratings = originalRatings.rdd
        .map(r => (r.getInt(ParameterTuningALS.indexMovieId_ratings), r.getDouble(ParameterTuningALS.indexRating_ratings)))

      // 3. Build mostRatedMovies that contains the 100 movies that were rated by the most users. This is very similar
      // to word-count, and finding the most frequent words in a document.
      val moviesAndRatingCounts = ratings.groupByKey()
        .map{ case (movieId, movieRatings) => (movieId, movieRatings.size) }
      val mostRatedMovies = sc.parallelize(movies.toList).join(moviesAndRatingCounts)
        .map{ case (movieId,(title, count)) => (movieId, title, count) }
        .sortBy({ case (_, _, count) => count }, ascending = false)
        .map{ case (movieId, title, _) => (movieId, title)}

      // Obtain selectedMovies List[(Int, String)] that contains 25 movies selected at random in mostRatedMovies as well
      // as their title. To select elements at random in a list, a good strategy is to shuffle the list (i.e. put it in
      // a random order) and take the first elements. Shuffling the list can be done with scala.util.Random.shuffle.
      val selectedMovies = shuffle(mostRatedMovies.take(100).toList).take(25)

      val userRatings = getRatings(selectedMovies)

      // 5. After finishing the rating, your program should display the top 20 movies that you might like. Look at the
      // recommendations, are you happy about your recommendations? Comment.
      // TODO
    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
  }

  // 4. You can now use your recommender system by executing the program you wrote! Write a function
  // getRatings(selectedMovies) gives you 25 movies to rate and you can answer directly in the console in the Scala IDE.
  // Give a rating from 1 to 5, or 0 if you do not know this movie.
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
