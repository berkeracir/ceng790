package edu.metu.ceng790.hw3

import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


object Credit {
  val MODEL_PATH = "model"

  // define the Credit Schema
  case class Credit(
    creditability: Double,
    balance: Double, duration: Double, history: Double, purpose: Double, amount: Double,
    savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double,
    residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double,
    credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double
  )
  // function to create a  Credit class from an Array of Double
  def parseCredit(line: Array[Double]): Credit = {
    Credit(
      line(0),
      line(1) - 1, line(2), line(3), line(4), line(5),
      line(6) - 1, line(7) - 1, line(8), line(9) - 1, line(10) - 1,
      line(11) - 1, line(12) - 1, line(13), line(14) - 1, line(15) - 1,
      line(16) - 1, line(17) - 1, line(18) - 1, line(19) - 1, line(20) - 1
    )
  }
  // function to transform an RDD of Strings into an RDD of Double
  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).map(_.map(_.toDouble))
  }

  def main(args: Array[String]) {

    val spark = SparkSession.builder.appName("Spark SQL").config("spark.master", "local[*]").getOrCreate()
    val sc = spark.sparkContext
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._
    // load the data into a  RDD
    val creditDF = parseRDD(sc.textFile("credit.csv")).map(parseCredit)
      .toDF("creditability", "balance", "duration", "history", "purpose", "amount", "savings", "employment",
        "instPercent", "sexMarried", "guarantors", "residenceDuration", "assets", "age", "concCredit", "apartment",
        "credits", "occupation", "dependents", "hasPhone", "foreign")
      .cache()

    // EXTRACT FEATURES
    // 1. Use a VectorAssembler to transform and return a new dataframe with all of the feature columns in a vector
    // column.
    val featureColumnNames = creditDF.columns.filter(colName => !colName.equals("creditability"))
    val featureColumnName = "features"

    val featuresAssembler = new VectorAssembler()
      .setInputCols(featureColumnNames)
      .setOutputCol(featureColumnName)

    // 2. Use a StringIndexer to return a Dataframe with the creditability column added as a label
    val labelColumnName = "label"

    val labelIndexer = new StringIndexer()
      .setInputCol("creditability")
      .setOutputCol(labelColumnName)
      .fit(creditDF)


    // 3. Use randomSplit function to split the data into two sets: 75% of the data is used to train (and tune) the
    // model, 25% will be used for testing.
    val Array(trainCreditDF, testCreditDF) = creditDF
      .randomSplit(Array(0.75, 0.25), seed = 4321)

    // TRAIN THE MODEL AND OPTIMIZE HYPERPARAMETERS
    // 4. Next, we train a RandomForest Classifier with the parameters:
    //    a. maxDepth: Maximum depth of a tree. Increasing the depth makes the model more powerful, but deep trees take
    //    longer to train.
    //    b. maxBins: Maximum number of bins used for discretizing continuous features and for choosing how to split on
    //    features at each node.
    //    c. impurity: Criterion used for information gain calculation
    //    d. auto: Automatically select the number of features to consider for splits at each tree node
    //    e. seed: Use a random seed number, allowing to repeat the results. Use the random seed 4321 in this
    //    assignment.
    // The model is trained by making associations between the input features and the labeled output associated with
    // those features. In order to find the best model, we search for the optimal combinations of the classifier
    // parameters.
    // You will optimize the model using a pipeline. A pipeline provides a simple way to try out different combinations
    // of parameters, using a process called grid search, where you set up the parameters to test, and MLLib will test
    // all the combinations. Pipelines make it easy to tune an entire model building workflow at once, rather than
    // tuning each element in the Pipeline separately.
    val randomForestClassifier = new RandomForestClassifier()
      .setFeaturesCol(featureColumnName)
      .setLabelCol(labelColumnName)

    // Use the ParamGridBuilder utility to construct the parameter grid with the following values: maxBins [24, 28, 32],
    // maxDepth, [3, 5, 7], impurity [“entropy", "gini"]
    val paramGridBuilder = new ParamGridBuilder()
      .addGrid(randomForestClassifier.featureSubsetStrategy, Array("auto"))
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .addGrid(randomForestClassifier.maxBins, Array(24, 28, 32))
      .addGrid(randomForestClassifier.maxDepth, Array(3, 5, 7))
      .build()

    // Additional ParamGridBuilders
    val extraParamGridBuilder = new ParamGridBuilder()
      .addGrid(randomForestClassifier.featureSubsetStrategy, Array("auto") ++ Array("all", "onethird", "sqrt", "log2"))
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .addGrid(randomForestClassifier.maxBins, Array(24, 28, 32) ++ Array(12, 16, 20, 36, 40))
      .addGrid(randomForestClassifier.maxDepth, Array(3, 5, 7) ++ Array(2, 4, 6, 8, 9, 10))
      .addGrid(randomForestClassifier.numTrees, Array(20, 24, 28, 32))
      .addGrid(randomForestClassifier.subsamplingRate, Array(0.1, 0.25, 0.5, 0.75, 1.0))
      .build()


    // 5. Next, you will create and set up a pipeline to make things easier. A Pipeline consists of a sequence of
    // stages, each of which is either an Estimator or a Transformer.
    // Use TrainValidationSplit that creates a (training, test) dataset pair. It splits the dataset into these two parts
    // using the trainRatio parameter. For example, with trainRatio=0.75, TrainValidationSplit will generate a training
    // and test dataset pair where 75% of the data is used for training and 25% for validation. Use these values in your
    // code. Note that, this is different from the original random data split. Here, we further divide the training
    // dataset into training and validation set, for tuning purposes. The final model that has been tuned will be used
    // to evaluate the result on the test set.
    val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol(labelColumnName)

    // The TrainValidationSplit uses an Estimator, a set of ParamMaps, and an Evaluator. Estimator should be your random
    // forest model, the ParamMaps is the parameter grid that you built in the previous step. The Evaluator should be
    // new BinaryClassificationEvaluator().
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(randomForestClassifier)
      .setEstimatorParamMaps(paramGridBuilder)
      .setEvaluator(binaryClassificationEvaluator)
      .setTrainRatio(0.75)
      .setSeed(4321)

    // Try CrossValidator
    val crossValidator = new CrossValidator()
      .setEstimator(randomForestClassifier)
      .setEstimatorParamMaps(paramGridBuilder)
      .setEvaluator(binaryClassificationEvaluator)
      .setNumFolds(2)
      .setSeed(4321)

    val pipeline = new Pipeline()
      .setStages(Array(featuresAssembler, labelIndexer, trainValidationSplit))
    //val pipeline = new Pipeline()
    //  .setStages(Array(featuresAssembler, labelIndexer, crossValidator))

    val model = pipeline.fit(trainCreditDF)
    model.write.overwrite().save(MODEL_PATH)

    val bestModel = model.stages(2).asInstanceOf[TrainValidationSplitModel]
      .bestModel.asInstanceOf[RandomForestClassificationModel]
    //val bestModel = model.stages(2).asInstanceOf[CrossValidatorModel]
    //  .bestModel.asInstanceOf[RandomForestClassificationModel]
    val impurity = bestModel.getImpurity
    val maxBins = bestModel.getMaxBins
    val maxDepth = bestModel.getMaxDepth
    println(s"""Model's Parameters => Impurity:\"$impurity\", MaxBins:$maxBins, MaxDepth:$maxDepth""")
//    val featureSubsetStrategy = bestModel.getFeatureSubsetStrategy
//    val numTrees = bestModel.getNumTrees
//    val subsamplingRate = bestModel.getSubsamplingRate
//    println(s"""Model's Parameters => FeatureSubsetStrategy:\"$featureSubsetStrategy\", Impurity:\"$impurity\" MaxBins:$maxBins, MaxDepth:$maxDepth, NumTrees:$numTrees, SubsamplingRate:$subsamplingRate""")

    // 6. Finally, evaluate the pipeline best-fitted model by comparing test predictions with test labels. You can use
    // transform function to get the predictions for test dataset. You can use evaluator’s evaluate function to get the
    // metrics.
    val trainPredictions = model.transform(trainCreditDF)
    val testPredictions = model.transform(testCreditDF)

    val trainResult = binaryClassificationEvaluator.evaluate(trainPredictions)
    val testResult = binaryClassificationEvaluator.evaluate(testPredictions)
    println(s"Model's Accuracies on =>  Train Data:$trainResult, Test Data:$testResult")
  }
}

