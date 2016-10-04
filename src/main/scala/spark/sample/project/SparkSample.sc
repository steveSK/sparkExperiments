import faker._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
val r = scala.util.Random
case class Person(firstName: String, surName: String, job: String, yearBorn: Integer)
def fakeEntry() : Person = {
  val name = Name.name.split(" ")
  return new Person(name(0), name(1), Company.name,  1950 + r.nextInt(50))

}
def repeat (times: Integer): List[_]  = {
  var result = List[Person]()
  for(i <- 1 to times) {
    result = fakeEntry() :: result
  }
  return result
}


val data = repeat(100)

val conf = new SparkConf().setAppName("test").setMaster("spark://stefan-Inspiron-7548:7077")
val sc = new SparkContext(conf)
val sqlContext = new SQLContext(sc)

val schema = StructType(Seq(StructField("firstName", StringType, false),
  StructField("surnName", StringType, false),
  StructField("job", StringType, false),
  StructField("yearBorn", IntegerType, false)
))



val rdd = sc.parallelize(data)
val dataDF = sqlContext.createDataFrame(rdd,classOf[Person])
dataDF.printSchema()
dataDF.registerTempTable("dataframe")

dataDF.rdd.getNumPartitions

val newDF = dataDF.distinct().select("*")

