package spark.sample.project

import faker.{Company, Name}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}
import spark.sample.utils.SparkConfig

import scala.util.Random

/**
  * Created by stefan on 10/3/16.
  */
object Spark_Dataframes_Simple {

  val  currentYear = 2016
  val  appName = "dataframes-simple"

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
    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMaster)
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val schema = StructType(Seq(StructField("fullName", StringType, false),
      StructField("job", StringType, false),
      StructField("yearBorn", IntegerType, false)
    ))

    val dataRows = data.map(v => Row(v.fullName, v.job, v.yearBorn))

    val rdd = sc.parallelize(dataRows)

    println("rdd created, lines: " + rdd.count())

    val dataDF = sqlContext.createDataFrame(rdd,schema)
    println("dataFrame created: " + dataDF.printSchema())
    dataDF.registerTempTable("dataframe")

    dataDF.rdd.getNumPartitions

    val uniqueDF = dataDF.distinct().select("*")
    uniqueDF.explain(true)

    val calculateAge: Integer => Integer = currentYear - _
    val myUdf = udf(calculateAge)

    val subDF = dataDF.withColumn("age", myUdf(dataDF.col("yearBorn"))).select("fullName", "job","age")
    subDF.explain(true)

    val filteredDF = subDF.filter(subDF("age") < 18)
    filteredDF.show()
    filteredDF.cache()
    val results = filteredDF.collect()

    subDF.orderBy(subDF("age").desc).show()
    subDF.groupBy().avg("age").show(false)

    println(filteredDF.count())

    val sampledDF = dataDF.sample(false, 0.20)
    sampledDF.show()

    filteredDF.unpersist()

  }

}
