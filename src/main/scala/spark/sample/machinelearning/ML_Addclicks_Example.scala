package spark.sample.machinelearning


import java.math.BigInteger

import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{lit, udf, when}
import org.apache.spark.sql.types._
import spark.sample.utils.SparkConfig

import scala.collection.mutable
import scala.collection.mutable.HashMap


/**
  * Created by stefan on 10/14/16.
  */
object ML_Addclicks_Example {


  val dataFilePath = "file:///home/stefan/dac_sample.txt"
  val epsilon = 1e-16
  val appName = "addClicksExample"


  def dotProduct(a: org.apache.spark.ml.linalg.Vector, b:org.apache.spark.ml.linalg.Vector) = {
    val listA = a.toArray.toList
    val listB = b.toArray.toList
    def dotProduct0(a: List[Double], b:List[Double], res: Double) : Double = {
      a match {
        case x::xs => dotProduct0(a.tail,b.tail,res + a.head * b.head )
        case Nil => res

      }
    }
    dotProduct0(listA,listB,0)
  }

  def sampleToRow(sample : List[(Int,String)]) : List[String] ={
     sample.map{ case(x,y) => y}
  }

  def oneHotEncoding(rawFeats : Seq[Row], oheDictBroadcast : scala.collection.Map[Row,Long], numOheFeats: Integer) : SparseVector = {
      val indexes = Array.fill[Int](rawFeats.size)(0)
      val values = Array.fill[Double](rawFeats.size)(0)
      for(el <- rawFeats) {
          indexes(rawFeats.indexOf(el)) = oheDictBroadcast.get(el).get.toInt
          values(rawFeats.indexOf(el)) = 1.0
      }
      Vectors.sparse(numOheFeats,indexes,values).asInstanceOf[SparseVector]
  }

  def oheUdfGenerator(ohe_dict_broadcast : Broadcast[scala.collection.Map[Row,Long]],numFeatures: Integer) = udf(
    (sample: Seq[Row]) => { oneHotEncoding(sample,ohe_dict_broadcast.value,numFeatures)

  })

  def createOneHotDict(dataFrame : Dataset[Row]) : scala.collection.Map[Row,Long] = {

    dataFrame.rdd.map(x => x.getSeq(0)).flatMap(x => x.asInstanceOf[Seq[Row]]).distinct().zipWithIndex().collectAsMap()
  }

  def parsePoint : Seq[String] => Seq[(Int,String)] = {
    (array) => { for {index <- 0 to array.length -1 if (array(index).nonEmpty)} yield ((index,array(index)))
    }
  }

  def logFunction1 : Double => Double = {
    -Math.log(_)

  }

  def logFunction0 : Double => Double = {
    (value) => {
      -Math.log(1 - value)
    }
  }

  def md5Hash(text: String) : String = java.security.MessageDigest.getInstance("MD5").digest(text.getBytes()).map(0xFF & _).map { Integer.toHexString(_) }.foldLeft(""){_ + _}


  def addLogLoss(df : Dataset[Row]) = {

    val udfLogFunction0 = udf(logFunction0)
    val udfLogFunction1 = udf(logFunction1)
    df.withColumn("log_loss", when(df("label") === 1,udfLogFunction1(df("p"))).when(df("label") === 0,udfLogFunction0(df("p")))).toDF()
  }

  def parseRawDF(rawDF : Dataset[Row],sparkSession: SparkSession): Dataset[Row] = {
    import sparkSession.implicits._
  //  val parsePointUdf = udf(parsePoint, ArrayType(StructType(Seq(StructField("_1", IntegerType, false),
  //  StructField("_2", StringType,false)))))
    val parsePointUdf = udf(parsePoint)


    val convertedRawDF = rawDF.rdd.map(x => x.getString(0).split("\t")).toDF()
    val getLabel: Seq[String] => Double = _(0).toDouble
    val getLabelUDF = udf(getLabel)

    val getFeatures: Seq[String] => Seq[String] = _.slice(1,7)
    val getFeaturesUDF = udf(getFeatures)

    return convertedRawDF.withColumn("label",getLabelUDF($"value"))
                    .withColumn("features",parsePointUdf(getFeaturesUDF($"value"))).drop("value")

  }

  def addProbability(df: Dataset[Row], model: LogisticRegressionModel, sparkSession: SparkSession): Dataset[Row] = {
    val coefficientsBroadcast = sparkSession.sparkContext.broadcast(model.coefficients)
    val intercept = model.intercept

    def getP: SparseVector => Double = (features: SparseVector) => {

        var rawPrediction = intercept + dotProduct(features,coefficientsBroadcast.value)
        rawPrediction = Math.min(rawPrediction, 20)
        rawPrediction = Math.max(rawPrediction, -20)
        Math.pow(1 + Math.exp(-rawPrediction),-1)
    }

    val getPUdf = udf(getP, DoubleType)
    df.withColumn("p", getPUdf(df("features")))
  }

  def evaluateResults(df : Dataset[Row], model: LogisticRegressionModel, sparkSession: SparkSession): Double = {
    val dfWithP = addProbability(df,model,sparkSession)
    val dfWithLogLoss = addLogLoss(dfWithP)
    return dfWithLogLoss.groupBy().avg("log_loss").take(1)(0).getDouble(0)
  }

  def hashFunction(rawFeats : List[(Int,String)], numBuckets : Int): HashMap[String,Int] = {

    val mapping = new mutable.HashMap[String,Int]()
    for(pair <- rawFeats) {
      val value = pair._2 + ":" + String.valueOf(pair._1)
        mapping.put(value,new BigInteger(md5Hash(value),16).intValue() % numBuckets)
    }
  /*  def map_update(l, r):
    l[r] += 1.0
    return l

    sparse_features = reduce(map_update, mapping.values(), defaultdict(float))
    return dict(sparse_features) */
    return mapping
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName(appName).setMaster(SparkConfig.sparkMaster)
    val sparkSession = SparkSession.builder().appName("test").master("spark://stefan-Inspiron-7548:7077").getOrCreate()
    sparkSession.sqlContext.setConf("spark.sql.shuffle.partitions", "6")
    import sparkSession.implicits._




    val sampleOne = List((0, "mouse"), (1
      , "black"))
    val sampleTwo = List((0, "cat"), (1, "tabby"), (2, "mouse"))
    val sampleThree =  List((0, "bear"), (1, "black"), (2, "salmon"))

    val schema = StructType(Seq(StructField("animal", StringType, false),
      StructField("color", StringType, false),
      StructField("food", StringType, true)
    ))

    val testVals = Array(sampleToRow(sampleOne), sampleToRow(sampleTwo), sampleToRow(sampleThree)).map(x => Row(x(0),x(1), if(x.size > 2) x(2) else null))
    val testRDD = sparkSession.sparkContext.parallelize(testVals)

    val testSampleDF = sparkSession.sparkContext.parallelize(Array(sampleOne, sampleTwo, sampleThree))

    val sampleOheDictManual = new HashMap[(Int,String),Int]
    sampleOheDictManual+= (0, "cat") -> 0
    sampleOheDictManual+= (0, "mouse") -> 1
    sampleOheDictManual+= (0, "bear") -> 2
    sampleOheDictManual+= (1, "black") -> 3
    sampleOheDictManual+= (1, "tabby") -> 4
    sampleOheDictManual+= (2, "mouse") -> 5
    sampleOheDictManual+= (2, "salmon") -> 6

    val aDense = Array(0.0, 3.0, 0.0, 4.0)
    val aSparse = Vectors.sparse(4,Array(1,3),Array(3.0,4.0))

    val bDense = Array(0.0, 0.0, 0.0, 1.0)
    val bSparse = Vectors.sparse(4,Array(3),Array(1.0))

    var array = Array.fill[Double](7)(0)
    array(sampleOheDictManual.get(0, "mouse").get) = 1.0
    array(sampleOheDictManual.get(1, "black").get) = 1.0



    val sampleOneOheFeatManual = Vectors.dense(array.clone())
    array = Array.fill[Double](7)(0)
    array(sampleOheDictManual.get(0, "cat").get) = 1.0
    array(sampleOheDictManual.get(1, "tabby").get) = 1.0
    array(sampleOheDictManual.get(2, "mouse").get) = 1.0
    val sampleTwoOheFeatManual = Vectors.dense(array.clone())
    array = Array.fill[Double](7)(0)
    array(sampleOheDictManual.get(0, "bear").get) = 1.0
    array(sampleOheDictManual.get(1, "black").get) = 1.0
    array(sampleOheDictManual.get(2, "salmon").get) = 1.0
    val  sampleThreeOheFeatManual = Vectors.dense(array.clone())





   // val schema2 = StructType(Seq(StructField("id",IntegerType,false),StructField("name",StringType,false)))
    val sampleDataDf = sparkSession.sparkContext.parallelize(Array(sampleOne,sampleTwo,sampleThree))

    val numSampleOheFeats = 7
    val sampleOheDictManualBroadcast = sparkSession.sparkContext.broadcast(sampleOheDictManual)

    //   Run one_hot_encoding() on sample_one.  Make sure to pass in the Broadcast variable.
/*    val sampleOneOheFeat = oneHotEncoding(sampleOne.toSeq,sampleOheDictManualBroadcast.value,numSampleOheFeats)
    println("sample_one_ohe: " + sampleOneOheFeat)
    val oheUdf = oheUdfGenerator(sampleOheDictManualBroadcast,numSampleOheFeats)
    val sampleOheDF = sampleDataDf.map(x => Row(oneHotEncoding(x,sampleOheDictManualBroadcast.value,numSampleOheFeats)))
    val schema2 = StructType(Seq(StructField("features", VectorType, true)))
    val result = sparkSession.createDataFrame(sampleOheDF,schema2)
    result.show() */
  //  sampleOheDF.collect()
  /*  val encoder = Encoders.tuple(Encoders.INT, Encoders.STRING)
    val distictFeatsSample = testSampleDF.flatMap(x => x)
    val distictFeatsSampleDF = distictFeatsSample.toDF("feature_id","category").distinct()
    distictFeatsSampleDF.show(false)

    val sampleOHEdict = distictFeatsSampleDF.rdd.zipWithIndex().collectAsMap() */
    val sampleOHEdict = createOneHotDict(testSampleDF.toDF())
    println(sampleOHEdict)


    val baseDF = sparkSession.read.text(dataFilePath).withColumnRenamed("value", "text")
    baseDF.show(false)

    val weights = Array(0.8, 0.1, 0.1)
    val seed = 42


    val listDFs =  baseDF.randomSplit(weights,seed).map(x => x.toDF())
    val trainDF = listDFs(0)
    trainDF.cache()
    val valDF = listDFs(1)
    valDF.cache()
    val testDF = listDFs(2)
    testDF.cache()


    baseDF.show()

    val parsedDF = parseRawDF(baseDF,sparkSession)
    parsedDF.show(false)
//    val parsedDFMore =  parsedDF.select("features").rdd.map(x => x.getSeq(0)).flatMap(x => x.asInstanceOf[Seq[(Int,String)]])
//    parsedDFMore.take(100).foreach(println)

    val ctrOheDict = createOneHotDict(parsedDF.select("features"))
    val numCtrOheFeats = ctrOheDict.count(x => true)
    println("number of entries: " + numCtrOheFeats)

    val oheDictBroadcast = sparkSession.sparkContext.broadcast(ctrOheDict)
    val oheDictUDF = oheUdfGenerator(oheDictBroadcast,numCtrOheFeats)
    val trainParsedDF =  parseRawDF(trainDF,sparkSession)
   // trainParsedDF.show(false)
   // val oheTrainDF = trainParsedDF.select($"label",oheDictUDF($"features"))
    trainParsedDF.printSchema()
    val encoder = Encoders.kryo[SparseVector]



    val oheTrainRDD = trainParsedDF.rdd.map(x => Row(x(0),oneHotEncoding(x.getSeq[Row](1),oheDictBroadcast.value,numCtrOheFeats)))
    val schemaTrain = StructType(Seq(StructField("label",DoubleType,false),StructField("features", VectorType, false)))
    val oheTrainDF = sparkSession.createDataFrame(oheTrainRDD,schemaTrain)
    oheTrainDF.show()

    /// .map{case Row(k: Int, v: String) => (k, v); case null => null}

    val standardization = false
    val elasticNetParam = 0.0
    val regParam = 0.01
    val maxIter = 20

    val lr = new LogisticRegression()
                 .setElasticNetParam(elasticNetParam)
                 .setRegParam(regParam)
                 .setMaxIter(maxIter)
                 .setStandardization(standardization)

    val lrModelBasic = lr.fit(oheTrainDF)

    println("intercept: " + lrModelBasic.intercept)
    println("length of coefficients: " + lrModelBasic.coefficients.size)

    val exampleLogLoss = Array((0.5, 1), (0.5, 0), (0.99, 1), (0.99, 0), (0.01, 1),
    (0.01, 0), (1.0, 1), (0.0, 1), (1.0, 0))
    val exampleLogLossDF = sparkSession.sparkContext.parallelize(exampleLogLoss).toDF("p","label")
    val logLossDF = addLogLoss(exampleLogLossDF)
    logLossDF.show()
    val classOneFracTrain = oheTrainDF.groupBy().avg("label")
    val oheTrainTempDF = oheTrainDF.withColumn("p",lit(classOneFracTrain.take(1)(0).getDouble(0)))
    val oheTrainLogLossDF = addLogLoss(oheTrainTempDF)

   // val addProbabilityModelBasic = addProbability(df, lr_model_basic)
    val trainingPredictions = addProbability(oheTrainDF,lrModelBasic,sparkSession).cache()

    trainingPredictions.show(1000)

    val logLossTrainModelBasic = evaluateResults(oheTrainDF, lrModelBasic,sparkSession)
    println("logLossAvg: " + logLossTrainModelBasic)


  }

}
