package spark.sample.project
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import spark.sample.project.Spark_Dataframes_Simple.appName
import spark.sample.utils.SparkConfig

import scala.util.matching.Regex

object ScalaDiamondsExample {

  def main(args: Array[String]): Unit ={

    val filePath = "/home/stefan/diamonds.csv"


    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMasterLocal)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()

    import sparkSession.implicits._


    val diamonds = sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema","true")
      .load(filePath)

    diamonds.show()
  }

}
