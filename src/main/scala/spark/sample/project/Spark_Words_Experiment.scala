package spark.sample.project

import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.udf
import spark.sample.utils.SparkConfig

import scala.collection.mutable

/**
  * Created by stefan on 10/3/16.
  */
object Spark_Words_Experiment {

  val appName = "words-experiment"
  val fileName = "file:///home/stefan/shakespear.txt"

  case class Word(word: String)
  case class WordCount(word: String, count: Int)


  def wordCount(wordListDF : DataFrame): DataFrame = {
    wordListDF.groupBy("word").count()
  }

  def detectIfBlank(word: String): Boolean = {
    word.trim.isEmpty
  }

  def removePunctuation(text: String): String = {
      return text.toLowerCase.replaceAll("""[\p{Punct}&&[^.]]""", "").replaceAll("""(?m)\s+$""", "").trim()
  }


  def main(args: Array[String]) {



    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMasterLocal)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()

    import sparkSession.implicits._


    val linesDataset = sparkSession.sparkContext.parallelize(Seq("Spark I am your father", "May the spark be with you", "Spark I am your father")).toDS()
    val wordsDataset = linesDataset.flatMap(x => x.split(" "))
    val removeDuplicates = wordsDataset.filter(_!="")


    val pairsDataset = removeDuplicates.map(w => (w,1))
      .withColumnRenamed("_1","word")
      .withColumnRenamed("_2","count")

    val pairsDatasetGrouped = pairsDataset.groupBy("word")
    val countDataset = pairsDatasetGrouped.count()
    countDataset.show()


    val pairsDataset0 = removeDuplicates.map(w => WordCount(w,1))

    val pairsDatasetGrouped0 = pairsDataset0.groupByKey(x => x.word)

    val countDataset0 = pairsDatasetGrouped0.reduceGroups((x,y) => WordCount(x.word,x.count + y.count)).toDF("key","count").select("count")
    countDataset0.show()

    val users = Seq((1, "user1"), (1, "user2"), (2, "user1"), (2, "user3"), (3,"user2"), (3,"user4"),(3,"user6"))

    // Input RDD
    val us = sparkSession.sparkContext.parallelize(users)


    val empty = Seq

    val res = us.mapValues(x => (x,"")).reduceByKey((x,y) => (x._1,y._1)).values
    res.collect().foreach(println)






    //1. task add s to column
    /*val addS: String => String = _.concat("s")
    val udf1 = udf(addS)
    val pluralDF = wordsDF.withColumn("word", udf1(wordsDF.col("word")))
    pluralDF.show()


    val func2: String => Integer = _.size
    val udf2 = udf(func2)
    val pluralLengthsDF = pluralDF.withColumn("word", udf2(pluralDF.col("word")))
    pluralLengthsDF.show()


    val wordCountsDF = wordsDF.groupBy("word").count()
    wordCountsDF.show()

    val uniqueWordsDF = wordsDF.distinct()
    println("unique words: " + uniqueWordsDF.count())

    val wordCountsDFMean = wordCountsDF.groupBy().mean("count")
    wordCountsDFMean.show()

    wordCount(wordsDF).show()
    val func3 = removePunctuation(_)
    val udf3 = udf(func3)

    val listSentences = List("Hi, you!", "No under_score!", "*      Remove punctuation then spaces  * '")
    val schemeSentences = StructType(Seq(StructField("sentence", StringType, false)))
    val dataSentencesRows = listSentences.map(x => Row(x))
    val rddSentences = sc.parallelize(dataSentencesRows)
    val sentenceDF = sqlContext.createDataFrame(rddSentences, schemeSentences)
    sentenceDF.show(false)
    sentenceDF.withColumn("sentence", udf3(sentenceDF.col("sentence"))).show(false)

    val shakespeareDFTemp = sqlContext.read.text(fileName)
    val shakespeareDF = shakespeareDFTemp.withColumn("value", udf3(shakespeareDFTemp.col("value")))
    shakespeareDF.show(15, false)

    val shakeWordsDF = shakespeareDF.explode(shakespeareDF.col("value")) {
      case Row(words: String) => words.split(" ").map(Word(_))
    }.select("word")
    shakeWordsDF.show()
    val func4 = detectIfBlank(_)
    val udf4 = udf(func4)
    val filteredShakeWordsDF = shakeWordsDF.filter(!udf4(shakeWordsDF.col("word")))
    filteredShakeWordsDF.show()


    val shakeWordCountDFTemp = wordCount(filteredShakeWordsDF)
    val shakeWordCountDF = shakeWordCountDFTemp.orderBy(shakeWordCountDFTemp("count").desc)
    shakeWordCountDF.show() */

  }
}
