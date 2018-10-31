package spark.sample.project

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.{Row, SparkSession}
import spark.sample.utils.SparkConfig

import scala.util.matching.Regex

/**
  * Created by stefan on 10/5/16.
  */
object Spark_Experiment_NASA_HTTP {

  val unknownLabel = "unknown"
  val fileName = "file:///home/stefan/nasa-http-logs"
  val appName = "nasa-logs-experiment"


  def useRegex(pattern: Regex) = udf(
    (string: String) => {
      val result = pattern.findAllIn(string).matchData

      if (result != null && !result.isEmpty) Some(result.toList(0).group(1))
      else None
    } match {
      case Some(s) => s
      case None => unknownLabel
    }

  )


  def main(args: Array[String]) {


    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMaster)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()


    val nasaHTTPlogsDF = sparkSession.read.text(fileName)
    val splitDF = nasaHTTPlogsDF.withColumn("host", useRegex(new Regex("^([^\\s]+\\s)"))(nasaHTTPlogsDF("value")))
      .withColumn("timestamp", useRegex(new Regex("^.*\\[(\\d\\d/\\w{3}/\\d{4}:\\d{2}:\\d{2}:\\d{2} -\\d{4})]"))(nasaHTTPlogsDF("value")))
      .withColumn("path", useRegex(new Regex("^.*\"\\w+\\s+([^\\s]+)\\s+HTTP.*\""))(nasaHTTPlogsDF("value")))
      .withColumn("status", useRegex(new Regex("^.*\"\\s+([^\\s]+)"))(nasaHTTPlogsDF("value")))
      .withColumn("content_size", useRegex(new Regex("^.*\\s+(\\d+)$"))(nasaHTTPlogsDF("value")))
      .drop(nasaHTTPlogsDF("value"))
    //  splitDF.persist()


    splitDF.printSchema()
    splitDF.show(false)
    import sparkSession.implicits._
    val clearedDF = sparkSession.createDataFrame(splitDF.rdd.map(row => {
      val colValue = row(4)
      if (colValue.equals(unknownLabel)) Row(row(0), row(1), row(2), row(3), "0") else row
    }), splitDF.schema)



     val badRowsDf =  clearedDF.filter( clearedDF("host").like("unknown") ||  clearedDF("timestamp").like(unknownLabel) ||
        clearedDF("path").like(unknownLabel) ||
        clearedDF("status").like(unknownLabel) ||
        clearedDF("content_size").like(unknownLabel))

      badRowsDf.printSchema()
      badRowsDf.explain()
      badRowsDf.show()
      println("null count: " + badRowsDf.show())

    val parsedLogsDF = clearedDF.withColumn("timestamp", unix_timestamp(clearedDF("timestamp"), "dd/MMM/yyyy:HH:mm:ss -SSS").cast("timestamp"))
    parsedLogsDF.cache()

    val contentSizeSummaryDF = parsedLogsDF.describe("content_size")
    contentSizeSummaryDF.show()


    val hostSumDF =(parsedLogsDF.groupBy("host").count())

    val hostMoreThan10DF = (hostSumDF
      .filter(hostSumDF("count") > 10).select(hostSumDF("host")))

    println("Any 20 hosts that have accessed more then 10 times:\n")
    hostMoreThan10DF.show(false)

    val not200DF = parsedLogsDF.filter(!parsedLogsDF("status").equalTo("200")).groupBy("path").count()
    val not200DFOrdered = not200DF.orderBy(not200DF("count").desc)

    println("totalCount: " + not200DFOrdered.count())
    not200DFOrdered.show(10,false)

    val uniqueHosts = parsedLogsDF.select("host").distinct()
    uniqueHosts.show(10,false)
    println("uniqueHosts: " + uniqueHosts.count())
    println("allCount: " + nasaHTTPlogsDF.count())


    val dayToHostPairDF = parsedLogsDF.withColumn("day", dayofmonth(parsedLogsDF("timestamp"))).select("host", "day")
    dayToHostPairDF.show()
    dayToHostPairDF.cache()

    val dayGroupHostsDF = dayToHostPairDF.groupBy("host").agg(collect_list("day") as "days")
    dayGroupHostsDF.show()

    val dailyHostsDF = dayToHostPairDF.distinct().groupBy("day").count().orderBy("day")
    dailyHostsDF.show(30,false)
    dailyHostsDF.cache()

    val totalReqPerDayDF = dayToHostPairDF.groupBy("day").count().withColumnRenamed("count","totalCount")

    val avgDaylyReqPerHostDFTemp = totalReqPerDayDF.join(dailyHostsDF,"day")
    val avgDaylyReqPerHostDF = avgDaylyReqPerHostDFTemp.withColumn ("avgCountPerHost",avgDaylyReqPerHostDFTemp("totalCount")/avgDaylyReqPerHostDFTemp("count"))
      .select("day","avgCountPerHost").orderBy("day")

    avgDaylyReqPerHostDF.show(30,false)

    val notFoundDF = parsedLogsDF.filter(parsedLogsDF("status").equalTo("404"))
    notFoundDF.show()
    notFoundDF.cache()

    val distinctPathsDF = notFoundDF.distinct().select("path")
    distinctPathsDF.show(40,false)

    val top20NotFoundDF = notFoundDF.groupBy("path").count().orderBy($"count".desc)
    top20NotFoundDF.show(20,false)


    val top25HostsNotFoundDF = notFoundDF.groupBy("host").count().orderBy($"count".desc)
    top25HostsNotFoundDF.show(25,false)

    val errorsByDateSortedDF = notFoundDF.withColumn("day", dayofmonth($"timestamp")).select("day").groupBy("day").count().orderBy("day")
    errorsByDateSortedDF.show()
    errorsByDateSortedDF.cache()

    val top5errorsByDateSortedDF = errorsByDateSortedDF.orderBy($"count".desc)
    top5errorsByDateSortedDF.show(5)

    val errorsByHourSortedDF = notFoundDF.withColumn("hour", hour($"timestamp")).select("hour").groupBy("hour").count().orderBy("hour")
    errorsByHourSortedDF.show(24)



    

  }
}
