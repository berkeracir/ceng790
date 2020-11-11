package edu.metu.ceng790.hw1

import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD._



object Part2 {
  def main(args: Array[String]): Unit = {
    var spark: SparkSession = null
    try {
      spark = SparkSession.builder().appName("Flickr using dataframes").config("spark.master", "local[*]").getOrCreate()
      val originalFlickrMeta: RDD[String] = spark.sparkContext.textFile("flickrSample.txt")
      
      // YOUR CODE HERE
      // 1. Display the 5 lines of the RDD (take(5)) and display the number of elements in the RDD (count()).
      originalFlickrMeta.take(5).foreach(println)
      println(originalFlickrMeta.count())

      // 2. Transform the RDD[String] in RDD[Picture] using the Picture class. Only keep interesting pictures having a
      // valid country and tags. To check your program, display 5 elements.
      val pictures = originalFlickrMeta.map(flickrMeta => new Picture(flickrMeta.split("\t")))
        .filter(picture => picture.hasValidCountry && picture.hasTags)
      pictures.take(5).foreach(println)

      // 3. Now group these images by country (groupBy). Print the list of images corresponding to the first country.
      // What is the type of this RDD?
      val picturesByCountries = pictures.groupBy(picture => picture.c)
      // Countries and their pictures count
      picturesByCountries.map(x => (x._1, x._2.size)).foreach(x => printf("%s(%d)\n", x._1, x._2))

      // 4. We now wish to process an RDD containing pairs in which the first element is a country, and the second
      // element is the list of tags used on pictures taken in this country. When a tag is used on multiple pictures, it
      // should appear multiple times in the list. As each image has its own list of tags, we need to concatenate these
      // lists, and the flatten function could be useful.
      val tagsByCountries = picturesByCountries.mapValues(pictures => pictures.flatMap(picture => picture.userTags))
      tagsByCountries.take(1).foreach(println)

      // 5. We wish to avoid repetitions in the list of tags, and would rather like to have each tag associated to its
      // frequency. Hence, we want to build a RDD of type RDD[(Country, Map[String, Int])]. The groupBy(identity)
      // function, equivalent to groupBy(x=>x) could be useful.
      val tagsFrequencyByCountries = tagsByCountries.mapValues(tags => tags.groupBy(tag => tag))
        .mapValues(pair => pair.map(p => (p._1, p._2.size)))
      tagsFrequencyByCountries.foreach(println)

      // 6. There are often several ways to obtain a result. The method we used to compute the frequency of tags in each
      // country quickly reaches a state in which the size of the RDD is the number of countries. This can limit the
      // parallelism of the execution as the number of countries is often quite small. Can you propose another way to
      // reach the same result without reducing the size of the RDD until the very end?

    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
    println("done")
  }
}