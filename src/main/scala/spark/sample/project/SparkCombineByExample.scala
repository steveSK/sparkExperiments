package spark.sample.project

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import spark.sample.project.Spark_Words_Experiment.appName
import spark.sample.utils.SparkConfig

import scala.collection.mutable

object SparkCombineByExample {

  def main(args: Array[String]): Unit ={

    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMasterLocal)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()

    import sparkSession.implicits._


    type ScoreCollector = (Int, Double)
    type PersonScores = (String, (Int, Double))

    val initialScores = Array(("Fred", 88.0), ("Fred", 95.0), ("Fred", 91.0), ("Wilma", 93.0), ("Wilma", 95.0), ("Wilma", 98.0))

    val wilmaAndFredScores = sparkSession.sparkContext.parallelize(initialScores).cache()

    val createScoreCombiner = (score: Double) => (1, score)

    val scoreCombiner = (collector: ScoreCollector, score: Double) => {
      val (numberScores, totalScore) = collector
      (numberScores + 1, totalScore + score)
    }

    val scoreMerger = (collector1: ScoreCollector, collector2: ScoreCollector) => {
      val (numScores1, totalScore1) = collector1
      val (numScores2, totalScore2) = collector2
      (numScores1 + numScores2, totalScore1 + totalScore2)
    }

    val scores = wilmaAndFredScores.combineByKey(createScoreCombiner, scoreCombiner, scoreMerger)


    val averagingFunction = (personScore: PersonScores) => {
      val (name, (numberScores, totalScore)) = personScore
      (name, totalScore / numberScores)
    }

    val personScores = scores.collectAsMap()
    personScores.foreach(println)

    val averageScores = scores.collectAsMap().map(averagingFunction)

    println("Average Scores using CombingByKey")
    averageScores.foreach((ps) => {
      val(name,average) = ps
      println(name+ "'s average score : " + average)
    })

    val keysWithValuesList = Array("foo=A", "foo=A", "foo=A", "foo=A", "foo=B", "bar=C", "bar=D", "bar=D")
    val data = sparkSession.sparkContext.parallelize(keysWithValuesList)
    //Create key value pairs
    val kv = data.map(_.split("=")).map(v => (v(0), v(1))).cache()

    val initialSet = mutable.HashSet.empty[String]
    val addToSet = (s: mutable.HashSet[String], v: String) => s += v
    val mergePartitionSets = (p1: mutable.HashSet[String], p2: mutable.HashSet[String]) => p1 ++= p2

    val uniqueByKey = kv.aggregateByKey(initialSet)(addToSet, mergePartitionSets)

    uniqueByKey.collect().foreach(println)

  }

}
