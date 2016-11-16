package spark.sample.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession

/**
  * Created by stefan on 11/12/16.
  */
object ML_NaiveBaies_Exer {

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
