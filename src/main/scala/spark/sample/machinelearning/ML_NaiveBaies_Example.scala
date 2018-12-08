package spark.sample.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import spark.sample.utils.{SparkConfig, SparkUtils}

/**
  * Created by stefan on 11/12/16.
  */
object ML_NaiveBaies_Example {

  val filepathTrain = "/home/stefan/adult-data-train.txt"
  val filepathTest = "/home/stefan/adult-data-test.txt"
  val appName = "naive-baies-example"

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkRemoteMaster)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()
    sparkSession.sqlContext.setConf("spark.sql.shuffle.partitions", "6")
    import sparkSession.implicits._

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

    val lambda = 1.0
    val modelType = "multinomial"

    val model = NaiveBayes.train(trainRDD, lambda, modelType)

    val labelAndPreds = testRDD.map { point =>
      val prediction = model.predict(org.apache.spark.mllib.linalg.Vectors.dense(point.features.toArray :+ 1.0))
      (point.label, prediction)
    }

    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val auROC = metrics.areaUnderROC()
    println("AREA under ROC: " + auROC)

  }


  }
