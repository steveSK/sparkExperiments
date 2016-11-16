package spark.sample.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.sql.SparkSession

/**
  * Created by stefan on 11/9/16.
  */
object ML_LogRegress_Exer {

  val filepathTrain = "/home/stefan/adult-data-train.txt"
  val filepathTest = "/home/stefan/adult-data-test.txt"

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("test").setMaster("spark://stefan-Inspiron-7548:7077")
    val sparkSession = SparkSession.builder().appName("test").master("spark://stefan-Inspiron-7548:7077").getOrCreate()
    sparkSession.sqlContext.setConf("spark.sql.shuffle.partitions", "6")
    import sparkSession.implicits._

    val trainRawDF = Utils.loadDataFromCSV(filepathTrain, sparkSession)
    val testRawDF = Utils.loadDataFromCSV(filepathTest, sparkSession)

    val continousColumns = List("education", "workclass", "marital-status",
      "occupation", "relationship", "race", "sex", "native-country")

    val toLabel: String => Double = x => x.trim match {
      case "<=50K." => 0.0
      case "<=50K" => 0.0
      case ">50K." => 1.0
      case ">50K" => 1.0
    }

    val trainRDD = Utils.transformRawDFToLabeledPointRDD(trainRawDF, continousColumns, toLabel, sparkSession)
    val testRDD = Utils.transformRawDFToLabeledPointRDD(testRawDF, continousColumns, toLabel, sparkSession)


    //logisticRegression
    val standardization = false
    val elasticNetParam = 0.0
    val regParam = 0.01
    val numIterations = 1000

    val lr = new LogisticRegressionWithLBFGS()
    lr.optimizer
      .setRegParam(regParam)
      .setNumIterations(numIterations)
      .setUpdater(new SimpleUpdater)

    val lrModel = lr.run(trainRDD)

    val lrScoreAndLabels = testRDD.map { point =>
      val score = lrModel.predict(org.apache.spark.mllib.linalg.Vectors.dense(point.features.toArray :+ 1.0))
      (score, point.label)
    }

    // Get evaluation metrics.
    val lrMetrics = new BinaryClassificationMetrics(lrScoreAndLabels)
    val auROC2 = lrMetrics.areaUnderROC()
    println("AREA under ROC: " + auROC2)
  }

}
