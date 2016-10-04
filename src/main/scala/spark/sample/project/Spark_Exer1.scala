package spark.sample.project

import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.udf

/**
  * Created by stefan on 10/3/16.
  */
object Spark_Exer1 {
  case class Word(word: String)

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

    val conf = new SparkConf().setAppName("test").setMaster("spark://stefan-Inspiron-7548:7077")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val list = List("cat", "elephant", "rat", "rat", "cat")
    val scheme = StructType(Seq(StructField("word", StringType, false)))
    val dataRows = list.map(x => Row(x))
    val rdd = sc.parallelize(dataRows)
    val wordsDF = sqlContext.createDataFrame(rdd, scheme)
    wordsDF.show()
    wordsDF.printSchema()

    //1. task add s to column
    val addS: String => String = _.concat("s")
    val udf1 = udf(addS)
    val pluralDF = wordsDF.withColumn("word", udf1(wordsDF.col("word")))
    pluralDF.show()


    val func2: String => Integer = _.size
    val udf2 = udf(func2)
    val pluralLengthsDF = pluralDF.withColumn("word", udf2(pluralDF.col("word")))
    pluralLengthsDF.show()


    //  val wordCountsDF = wordsDF.groupBy("word").count()
    //  wordCountsDF.show()

    val uniqueWordsDF = wordsDF.distinct()
    println("unique words: " + uniqueWordsDF.count())

    //  val wordCountsDFMean = wordCountsDF.groupBy().mean("count")
    //  wordCountsDFMean.show()

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


    val fileName = "file:///home/stefan/shakespear.txt"

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
    shakeWordCountDF.show()

  }
}
