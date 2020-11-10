package edu.metu.ceng790.hw1

import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{Encoders, Row, SaveMode, SparkSession}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.ByteType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.FloatType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.LongType
import java.net.URLDecoder

import org.apache.spark.sql.catalyst.dsl.expressions.{DslExpression, StringToAttributeConversionHelper}

import scala.tools.scalap.scalax.rules.scalasig.ClassFileParser.header

object Part1 {
  def main(args: Array[String]): Unit = {

    var spark: SparkSession = null
    try {
      
      spark = SparkSession.builder()
        .appName("Flickr using dataframes")
        .config("spark.master", "local[*]")
        .getOrCreate()

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
      originalFlickrMeta.createOrReplaceTempView("originalSamples")

      // 1. Using the Spark SQL API (accessible with spark.sql("...")), select fields containing the identifier, GPS
      // coordinates, and type of license of each picture.
      val samplesWithReducedColumns = spark
        .sql("SELECT photo_id, longitude, latitude, license FROM originalSamples WHERE marker = 0")

      // 2. Create a DataFrame containing only data of interesting pictures, i.e. pictures for which the license
      // information is not null, and GPS coordinates are valid (not -1.0).
      val interestingPictures = samplesWithReducedColumns
        .filter("longitude <> -1.0 AND latitude <> -1.0 AND license <> ''")
      interestingPictures.createOrReplaceTempView("interestingPictures")

      // 3. Display the execution plan used by Spark to compute the content of this DataFrame(explain()).
      println("### Explain Interesting Pictures DataFrame:\n")
      interestingPictures.explain(true)

      // 4. Display the data of this pictures (show()). Keep in mind that Spark uses lazy execution, so as long as we do
      // not perform any action, the transformations are not executed.
      interestingPictures.show(false)

      // 5. Our goal is now to select the pictures whose license is NonDerivative. To this end we will use a second file
      // containing the properties of each license. Load this file in a DataFrame and do a join operation to identify
      // pictures that are both interesting and NonDerivative. Examine the execution plan and display the results.
      val originalFlickrLicenseMeta = spark.read
        .format("csv")
        .option("delimiter", "\t")
        .option("header", "true")
        .load("FlickrLicense.txt")
      originalFlickrLicenseMeta.createOrReplaceTempView("licenses")

      val interestingAndNonDerivativeLicencedPictures = spark
        .sql("SELECT interestingPictures.* FROM interestingPictures " +
          "INNER JOIN licenses ON licenses.NonDerivative = 1 AND interestingPictures.license=licenses.name")

      println("### Explain Interesting And NonDerivative Licensed Pictures DataFrame:\n")
      interestingAndNonDerivativeLicencedPictures.explain(true)
      interestingAndNonDerivativeLicencedPictures.show(false)

      // 6. During a work session, it is likely that we reuse multiple time the DataFrame of interesting pictures. It
      // would be a good idea to cache it to avoid recomputing it from the file each time we use it. Do this, and
      // examine the execution plan of the join operation again. What do you notice?
      interestingAndNonDerivativeLicencedPictures.cache()
      println("### Explain Interesting And NonDerivative Licensed Pictures DataFrame after cached:\n")
      interestingAndNonDerivativeLicencedPictures.explain(true)

      // 7. Save the final result in a csv file (write). Don’t forget to add a header to reuse it more easily.
      interestingAndNonDerivativeLicencedPictures.coalesce(1).write
        .mode(SaveMode.Overwrite)
        .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false")
        .option("delimiter", "\t")
        .option("header", "true")
        .csv("part1_out")

    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
    println("done")
  }
}