package spark.sample.sparkstreaming

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StringType, StructField, StructType, TimestampType}
import spark.sample.utils.SparkConfig
import org.apache.spark.sql.functions._

object SparkStreamingExample {

  val filePath = "/home/stefan/events.json"
  val appName = "streamingApp"

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMasterLocal)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()

    import sparkSession.implicits._

    val jsonSchema = StructType(Seq(StructField("time", TimestampType, true),
      StructField("action", StringType, true)))

    val streamingDS = sparkSession.readStream.option("maxFilesPerTrigger", 1).schema(jsonSchema).json(filePath)

    val streamingCountDS = streamingDS
      .withWatermark("time", "2 minutes")
      .groupBy($"action",window($"time", "1 hour"))
      .count()

    println(streamingCountDS.isStreaming)

    val query =
      streamingCountDS
        .writeStream
        .format("json")        // memory = store in-memory table (for testing only in Spark 2.0)
        .option("basePath", "/home/stefan/spark-streaming/")
        .option("path", "/home/stefan/spark-streaming/")
        .option("checkpointLocation", "/home/stefan/spark-streaming/")// counts = name of the in-memory table// complete = all the counts should be in the table
        .start()


    while(true){}


  }

}
