package spark.sample.project

import faker.{Company, Name}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

/**
  * Created by stefan on 10/3/16.
  */
object Spark_Less1 {

  val  CURRENT_YEAR = 2016

  case class Person(fullName: String, job: String, yearBorn: Integer)
  def fakeEntry() : Person = {
    val r = new Random()
    return new Person(Name.name, Company.name.filterNot(x => x==','),  1950 + r.nextInt(50))

  }
  def repeat (times: Integer): List[Person]  = {
    var result = List[Person]()
    for(i <- 1 to times) {
      result = fakeEntry() :: result
    }
    return result
  }

  def main(args: Array[String]) {
    val data  = repeat(100)
    print(data)
    val conf = new SparkConf().setAppName("test").setMaster("spark://stefan-Inspiron-7548:7077")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val schema = StructType(Seq(StructField("fullName", StringType, false),
      StructField("job", StringType, false),
      StructField("yearBorn", IntegerType, false)
    ))

    val dataRows = data.map(v => Row(v.fullName, v.job, v.yearBorn))



    val rdd = sc.parallelize(dataRows)
   // val rddRow = rdd.map(v => Row(v: _*))

    println("rdd created, lines: " + rdd.count())

    val dataDF = sqlContext.createDataFrame(rdd,schema)
    println("dataFrame created: " + dataDF.printSchema())
    dataDF.registerTempTable("dataframe")

    dataDF.rdd.getNumPartitions

   // val newDF = dataDF.distinct().select("*")
   // newDF.explain(true)

    val calculateAge: Integer => Integer = CURRENT_YEAR - _
    val myUdf = udf(calculateAge)

    val subDF = dataDF.withColumn("age", myUdf(dataDF.col("yearBorn"))).select("fullName", "job","age")
    //val subDF = dataDF.select("fullName", "job","yearBorn")
    subDF.explain(true)
    val filteredDF = subDF.filter(subDF("age") < 18)
    filteredDF.cache()
    //val results = filteredDF.collect()
    //filteredDF.show()
    subDF.orderBy(subDF("age").desc).show()
   // subDF.groupBy().avg("age").show(false)
    filteredDF.count()
    val sampledDF = dataDF.sample(false, 0.20)
    sampledDF.show()

    filteredDF.unpersist()

  }

}
