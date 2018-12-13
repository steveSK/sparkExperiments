package spark.sample.project

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import spark.sample.utils.SparkConfig

object SparkWikiExample {

  val  appName = "wiki-pagecount"
  val pageCountFile = "file:///home/stefan/pagecounts"

  case class PageView(domainCode: String, pageTitle: String,  countViews: Option[Int], responseSize: Option[Int])


  def main(args: Array[String]) {
    //print(data)
    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMasterLocal)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()

    import sparkSession.implicits._

   // val pagecounts = sparkSession.sparkContext.textFile(pageCountFile)
    val pageCountsDataset = sparkSession.read
      .option("delimiter", " ")
      .option("inferSchema","true")
      .csv(pageCountFile)
      .toDF("domainCode","pageTitle","countViews","responseSize")
      .withColumn("countViews", 'countViews.cast(IntegerType))
      .withColumn("responseSize", 'responseSize.cast(IntegerType)).as[PageView]

    pageCountsDataset.show()

    val enPages = pageCountsDataset.filter(x => x.domainCode == "en")
    enPages.printSchema()
    val enPagesOver = enPages.filter(x => x.countViews.getOrElse(0) > 20000)

    enPagesOver.show()



  }

}
