package spark.sample.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.sql.SparkSession
import spark.sample.utils.{SparkConfig, SparkUtils}

/**
  * Created by stefan on 11/4/16.
  */
object ML_SVM_Example {

  val filepathTrain = "/home/stefan/adult-data-train.txt"
  val filepathTest = "/home/stefan/adult-data-test.txt"
  val appName = "SVM-example"

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMaster)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()
    sparkSession.sqlContext.setConf("spark.sql.shuffle.partitions", "6")

    val trainRawDF = SparkUtils.loadDataFromCSV(filepathTrain, sparkSession)
    val testRawDF = SparkUtils.loadDataFromCSV(filepathTest, sparkSession)

    val continousColumns = List("education", "workclass", "marital-status",
      "occupation", "relationship", "race", "sex", "native-country")

    val toLabel: String => Double = x => x.trim match {
      case "<=50K." => 0.0
      case "<=50K" => 0.0
      case ">50K." => 1.0
      case ">50K" => 1.0
    }

    val trainRDD = SparkUtils.transformRawDFToLabeledPointRDD(trainRawDF, continousColumns, toLabel, sparkSession)
    val testRDD = SparkUtils.transformRawDFToLabeledPointRDD(testRawDF, continousColumns, toLabel, sparkSession)

    val numIterations = 1000
    val regParam = 0.01
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .setStepSize(1)
      .setUpdater(new L1Updater)

    val svmModel = svmAlg.run(trainRDD)



    svmModel.clearThreshold()

    // Compute raw scores on the test set.
    val svmScoreAndLabels = testRDD.map { point =>
      val score = svmModel.predict(org.apache.spark.mllib.linalg.Vectors.dense(point.features.toArray :+ 1.0))
      (score, point.label)
    }

    // Get evaluation metrics.
    val svmMetrics = new BinaryClassificationMetrics(svmScoreAndLabels)
    val auROC1 = svmMetrics.areaUnderROC()
    println("AREA under ROC: " + auROC1)
  }

}
