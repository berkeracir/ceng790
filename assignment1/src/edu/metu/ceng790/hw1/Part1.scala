package edu.metu.ceng790.hw1

import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.ByteType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.FloatType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.LongType
import java.net.URLDecoder
import org.apache.spark.sql.Row
import org.apache.spark.sql.Encoders

object Part1 {
  def main(args: Array[String]): Unit = {

    var spark: SparkSession = null
    try {
      
      spark = SparkSession.builder().appName("Flickr using dataframes").config("spark.master", "local[*]").getOrCreate()

      //   * Photo/video identifier
      //   * User NSID
      //   * User nickname
      //   * Date taken
      //   * Date uploaded
      //   * Capture device
      //   * Title
      //   * Description
      //   * User tags (comma-separated)
      //   * Machine tags (comma-separated)
      //   * Longitude
      //   * Latitude
      //   * Accuracy
      //   * Photo/video page URL
      //   * Photo/video download URL
      //   * License name
      //   * License URL
      //   * Photo/video server identifier
      //   * Photo/video farm identifier
      //   * Photo/video secret
      //   * Photo/video secret original
      //   * Photo/video extension original
      //   * Photos/video marker (0 = photo, 1 = video)

      val customSchemaFlickrMeta = StructType(Array(
        StructField("photo_id", LongType, true),
        StructField("user_id", StringType, true),
        StructField("user_nickname", StringType, true),
        StructField("date_taken", StringType, true),
        StructField("date_uploaded", StringType, true),
        StructField("device", StringType, true),
        StructField("title", StringType, true),
        StructField("description", StringType, true),
        StructField("user_tags", StringType, true),
        StructField("machine_tags", StringType, true),
        StructField("longitude", FloatType, false),
        StructField("latitude", FloatType, false),
        StructField("accuracy", StringType, true),
        StructField("url", StringType, true),
        StructField("download_url", StringType, true),
        StructField("license", StringType, true),
        StructField("license_url", StringType, true),
        StructField("server_id", StringType, true),
        StructField("farm_id", StringType, true),
        StructField("secret", StringType, true),
        StructField("secret_original", StringType, true),
        StructField("extension_original", StringType, true),
        StructField("marker", ByteType, true)))

      val originalFlickrMeta = spark.sqlContext.read
        .format("csv")
        .option("delimiter", "\t")
        .option("header", "false")
        .schema(customSchemaFlickrMeta)
        .load("flickrSample.txt")
        
        // YOUR CODE HERE
        
    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
    println("done")
  }
}