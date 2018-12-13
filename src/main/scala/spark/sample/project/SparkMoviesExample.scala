package spark.sample.project


import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, BooleanType, IntegerType, StringType}
import spark.sample.utils.SparkConfig
import org.apache.spark.sql.functions._

object SparkMoviesExample {

  val filePathMovies = "/home/stefan/titles.csv";
  val filePathActors = "/home/stefan/actors.csv";
  val filePathRatings = "/home/stefan/ratings.csv"
  val appName = "moviesApp"


  case class Actor(id:String, primaryName:String ,birthYear:Int, deathYear:Int, primaryProfession: String, knownForTitles:Array[String])
  case class Movie(id:String, titleType: String, primaryTitle: String, originalTitle: String, isAdult: Boolean, startYear: Int, endYear: Int, runtimeMinutes: Int, genres: String)
  case class Rating(id:String, averageRating: Double, numVotes: Int)



  def main(args: Array[String]): Unit ={

    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMasterLocal)
    val sparkSession = SparkSession.builder().config(conf).getOrCreate()

    import sparkSession.implicits._

    val actorsDataset = sparkSession.read
      .option("sep", "\t")
      .option("inferSchema","true")
      .option("header","true")
      .csv(filePathActors)
      .withColumnRenamed("nconst","id")
      .withColumn("birthYear", 'birthYear.cast(IntegerType))
      .withColumn("deathYear", 'deathYear.cast(IntegerType))
      .withColumn("knownForTitles",split($"knownForTitles",",").cast(ArrayType(StringType)))
      .as[Actor]

    val moviesDataset = sparkSession.read
      .option("sep", "\t")
      .option("inferSchema","true")
      .option("header","true")
      .csv(filePathMovies)
      .withColumnRenamed("tconst","id")
      .withColumn("isAdult",'isAdult.cast(BooleanType))
      .withColumn("startYear", 'startYear.cast(IntegerType))
      .withColumn("endYear", 'endYear.cast(IntegerType))
      .withColumn("runtimeMinutes",'runtimeMinutes.cast(IntegerType))
      .as[Movie]


    val ratingsDataset = sparkSession.read
      .option("sep", "\t")
      .option("inferSchema","true")
      .option("header","true")
      .csv(filePathRatings)
      .withColumnRenamed("tconst","id")
      .withColumn("numVotes",'numVotes.cast(IntegerType))
      .as[Rating]


    actorsDataset.printSchema()
    actorsDataset.show()

    //val moviesWithRatings = moviesDataset.join(ratingsDataset,Seq("id"),"inner")

    //val topTenMovies = moviesWithRatings.filter($"numVotes">100000).filter($"averageRating">8.5).orderBy(desc("averageRating")).limit(10)

    //topTenMovies.show()

    val moviesByActor = actorsDataset.withColumn("id", explode($"knownForTitles")).join(moviesDataset,Seq("id"))
    val bradPitMovies = moviesByActor.filter($"primaryName"==="Brad Pitt")




    val toHours = (x:Int) => x.toDouble/60
    val toHoursUdf = udf(toHours)

    val movieDatasetUpdated = moviesDataset.withColumn("runtimeMinutes",bround(toHoursUdf($"runtimeMinutes"),2)).withColumnRenamed("runtimeMinutes","runtime")

    val screenTime = moviesByActor.groupBy($"primaryName").sum("runtimeMinutes")
      .withColumnRenamed("sum(runtimeMinutes)","screenTime")
      .withColumn("screenTime", when($"screenTime".isNull, 0).otherwise($"screenTime"))
      .orderBy(desc("screenTime"))
    screenTime.show()


  }


}
