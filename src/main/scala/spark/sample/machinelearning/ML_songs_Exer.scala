package spark.sample.machinelearning

import org.apache.spark.SparkConf
import breeze.linalg.*
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, PolynomialExpansion}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

/**
  * Created by stefan on 10/12/16.
  */
object ML_songs_Exer {

  val songFilePath = "/home/stefan/millionsong.txt"


  val convertToLabeledPoints: String => LabeledPoint = x => {
     val list = x.split(',').map(x => x.toDouble)
     LabeledPoint(list.head, Vectors.dense(list.tail))
  }

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

  def vecAdd(a: org.apache.spark.ml.linalg.Vector, b:org.apache.spark.ml.linalg.Vector) = {
    val listA = a.toArray.toList
    val listB = b.toArray.toList
    def dotProduct0(a: List[Double], b:List[Double], res: List[Double]) : org.apache.spark.ml.linalg.Vector = {
      a match {
        case x::xs => dotProduct0(a.tail,b.tail,(a.head + b.head) :: res)
        case Nil => Vectors.dense(res.toArray)

      }
    }
    dotProduct0(listA,listB,Nil)
  }

  def vecSub(a: org.apache.spark.ml.linalg.Vector, b:org.apache.spark.ml.linalg.Vector) = {
    val listA = a.toArray.toList
    val listB = b.toArray.toList
    def dotProduct0(a: List[Double], b:List[Double], res: List[Double]) : org.apache.spark.ml.linalg.Vector = {
      a match {
        case x::xs => dotProduct0(a.tail,b.tail,(a.head - b.head) :: res)
        case Nil => Vectors.dense(res.toArray)

      }
    }
    dotProduct0(listA,listB,Nil)
  }

  def multiplyByScalar(scalar : Double,v: org.apache.spark.ml.linalg.Vector) = {
    val listV = v.toArray.toList
    def multiplyByScalar0(v: List[Double],res : List[Double]) : org.apache.spark.ml.linalg.Vector={
      v match {
        case x::xs => multiplyByScalar0(xs,scalar*x::res)
        case Nil => Vectors.dense(res.toArray)
      }
    }
    multiplyByScalar0(listV,List())
  }

  def zeroVector(l : Integer): Array[Double] ={
    var result = new Array[Double](l)
    var i = 0
    while(i < l){
      result(i) = 0.0
      i = i + 1
    }
    return result
    }


  def getCombinations(list: Array[Double]) : Array[Double] ={
    val res = (for{ x <- list;
          y <- list} yield (x*y))

    return list ++ res

  }



  def  twoWayInteractions(lp : LabeledPoint): LabeledPoint  = {
      LabeledPoint(lp.label,Vectors.dense(getCombinations(lp.features.toArray)))
  }



  def gradientSummand(weights : org.apache.spark.ml.linalg.Vector, lp: LabeledPoint) = {
   multiplyByScalar((dotProduct(weights,lp.features) - lp.label),lp.features)
  }

  def getLabeledPrediction(weights: Array[Double], lp : LabeledPoint): (Double,Double) ={
      return (dotProduct(Vectors.dense(weights),lp.features),lp.label)
  }

  def linregGradientDescent(trainData : RDD[LabeledPoint], numIters : Integer, sparkSession: SparkSession) : (Array[Double],Array[Double]) = {
    import sparkSession.implicits._
    // The length of the training data
      val n = trainData.count()
      val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label")
      .setPredictionCol("prediction")
    // The number of features in the training data
      val d = trainData.first().features.size

    var w = zeroVector(d)
    println(w.toString)
    val alpha = 1.0
    // We will compute and store the training error after each iteration
    val errorTrain = zeroVector(numIters)
    for(i <- 0 to numIters - 1){

    // Use get_labeled_prediction from (3b) with trainData to obtain an RDD of (label, prediction)
    //tuples.  Note that the weights all equal 0 for the first iteration, so the predictions will
    // have large errors to start.
      val predAndLabelsTrain = trainData.map(x => getLabeledPrediction(w,x))
      val predAndLabelsTrainDF = sparkSession.createDataset(predAndLabelsTrain).toDF("prediction","label")
   //   predAndLabelsTrainDF.show()
      errorTrain(i) = evaluator.evaluate(predAndLabelsTrainDF)

    //  Calculate the `gradient`.  Make use of the `gradient_summand` function you wrote in (3a).
 //     Note that `gradient` should be a `DenseVector` of length `d`.
       val gradient : org.apache.spark.ml.linalg.Vector  = trainData.map(x => gradientSummand(Vectors.dense(w),x)).reduce((x1,x2) => vecAdd(x1,x2))

      //.aggregate(Vectors.dense(zeroVector(d)))((acc, value) => (vecAdd(acc,value)),((acc1:org.apache.spark.ml.linalg.Vector, acc2:org.apache.spark.ml.linalg.Vector) => vecAdd(acc1,acc2)))

      //Update the weights
        val alphaI = alpha / (n * Math.sqrt(i+1))
        val toSubtr = multiplyByScalar(alphaI,gradient)
        w = vecSub(Vectors.dense(w),toSubtr).toArray


    }
    return (w,errorTrain)

  }





  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("spark://stefan-Inspiron-7548:7077")
    val sparkSession = SparkSession.builder().appName("test").master("spark://stefan-Inspiron-7548:7077").getOrCreate()
    import sparkSession.implicits._


    val baseDF = sparkSession.read.text(songFilePath)


    val udf1 = udf(convertToLabeledPoints)
    val labeledDF = baseDF.withColumn("value",udf1($"value"))
   // labeledDF.show(false)
    labeledDF.cache()

    val minYear = labeledDF.selectExpr("min(value.label)").take(1)(0).getDouble(0)
    val maxYear = labeledDF.selectExpr("max(value.label)").take(1)(0).getDouble(0)
    println("min: " + minYear)
    println("max: " + maxYear)
    //val yearRange = maxYear - minYear
    val parsedDF = labeledDF.select($"value.label" - minYear as "label",$"value.features".as("features"))
   // val parsedDF = parsedDFTemp.withColumn("label",$"label" - minYear)
  //  parsedDF.show(false)

    val seed = 42
    val weights = Array(0.8,0.1,0.1)

    val listDFs =  parsedDF.randomSplit(weights).map(x => x.toDF())
    val trainDF = listDFs(0)
    val valDF = listDFs(1)
    val testDF = listDFs(2)

    //println("totalCount: " + (trainDF.count() + valDF.count() + testDF.count()))


    val avgTrainYear = trainDF.selectExpr("avg(label)").first().getDouble(0)
    println("avg: " + avgTrainYear)

    val predsAndLabels = Array((1.0, 3.0), (2.0, 1.0), (2.0, 2.0))
    val testSchema = StructType(Seq(StructField("prediction", DoubleType, false),
      StructField("label", DoubleType, false)
    ))
    // cool way how to create DataFrame from Array
    val predsAndLabelsDF = sparkSession.sparkContext.parallelize(predsAndLabels).toDF("prediction","label")

    val evaluator = new RegressionEvaluator()
          .setMetricName("rmse")
          .setLabelCol("label")
          .setPredictionCol("prediction")

    val value = evaluator.evaluate(predsAndLabelsDF)
    println("val: " + value)

    val predAndLabelsTrain = trainDF.drop("features").withColumn("prediction",$"label" * 0.0 + avgTrainYear)
    val rmseTrainBase = evaluator.evaluate(predAndLabelsTrain)

    val predAndLabelsVal = valDF.drop("features").withColumn("prediction",$"label" * 0.0 + avgTrainYear)
    val rmseValBase = evaluator.evaluate(predAndLabelsVal)

    val predAndLabelsTest = testDF.drop("features").withColumn("prediction",$"label" * 0.0 + avgTrainYear)
    val rmseTestBase = evaluator.evaluate(predAndLabelsTest)

    println("rmseTrain " + rmseTrainBase)
    println("rmseVal " + rmseValBase)
    println("rmseTest " + rmseTestBase)


      var exampleW = new DenseVector(Array(1.0, 1.0, 1.0))
      var exampleLP = LabeledPoint(2.0, Vectors.dense(Array(3.0, 1.0, 4.0)))
      val summandOne = gradientSummand(exampleW, exampleLP)


      exampleW = new DenseVector(Array(0.24, 1.2, -1.4))
      exampleLP = LabeledPoint(3.0, Vectors.dense(Array(-1.4, 4.2, 2.1)))
      val summandTwo = gradientSummand(exampleW, exampleLP)
      println("summandOne " + summandOne)
      println("summandTwo " + summandTwo)


    val weights2 = Array(1.0, 1.5)
    val predictionExample = sparkSession.sparkContext.parallelize(Array(LabeledPoint(2.0, Vectors.dense(Array(1.0, 0.5))),
    LabeledPoint(1.5, Vectors.dense(Array(0.5, 0.5)))))

    val predAndLabelsExample = predictionExample.map(x => getLabeledPrediction(weights2,x))
    predAndLabelsExample.collect().foreach(println)


    val exampleN = 10
    val exampleD = 3
    val exampleDataTemp = trainDF.orderBy($"label".desc).take(exampleN)
    exampleDataTemp.foreach(println)


    val exampleData = sparkSession.sparkContext.parallelize(exampleDataTemp.map(x => LabeledPoint(x(0).asInstanceOf[Double],
      Vectors.dense(x(1).asInstanceOf[org.apache.spark.ml.linalg.Vector].toArray.take(exampleD)))))
    val numIters1 = 5
    val result1 = linregGradientDescent(exampleData, numIters1,sparkSession)
    println("weight: " + result1._1.foreach(println))
    println("error: " + result1._2.foreach(println))


    val numIters2 = 50
    val result2 = linregGradientDescent(trainDF.map(x => LabeledPoint(x(0).asInstanceOf[Double],
      Vectors.dense(x(1).asInstanceOf[org.apache.spark.ml.linalg.Vector].toArray))).rdd,numIters2,sparkSession)
    val finalWeights = result2._1
    val finalErrors = result2._2
    val finalPredsAndLabels = parsedDF.map(x => getLabeledPrediction(finalWeights,LabeledPoint(x(0).asInstanceOf[Double],
      x(1).asInstanceOf[org.apache.spark.ml.linalg.Vector])))

    val finalPredsAndLabelsDF = finalPredsAndLabels.toDF("prediction","label")

    val finalRMSE = evaluator.evaluate(finalPredsAndLabelsDF)

    finalPredsAndLabelsDF.show()
    finalWeights.foreach(println)



    val numIters = 500  // iterations
    val reg = 1e-1  // regParam
    val alpha = 0.2  // elasticNetParam
    val useIntercept = true


    // TODO: Replace <FILL IN> with appropriate code
      val linReg1 = new LinearRegression().setMaxIter(numIters)
                                         .setRegParam(reg)
                                         .setElasticNetParam(alpha)
                                         .setFitIntercept(useIntercept)

      val firstModel = linReg1.fit(trainDF)

    // coeffsLR1 stores the model coefficients; interceptLR1 stores the model intercept
        val coeffsLR1 = firstModel.coefficients
        val interceptLR1 = firstModel.intercept

        println(coeffsLR1)
        println(interceptLR1)

    val samplePrediction = firstModel.transform(trainDF)
    samplePrediction.show()

    val rmse = evaluator.evaluate(samplePrediction)
    println("rmse: " + rmse)


    var bestRMSE = rmse
    var bestRegParam = reg
    var bestModel = firstModel


    for(regParam <- Array(1e-1,1e-10,1e-5,1.0)) {
      val linReg = new LinearRegression().setMaxIter(numIters)
        .setRegParam(regParam)
        .setElasticNetParam(alpha)
        .setFitIntercept(useIntercept)

      val model = linReg.fit(trainDF)
      val predDF = model.transform(valDF)

      val rmseValGrid = evaluator.evaluate(predDF)
      println("rsme: " + rmseValGrid)

      if (rmseValGrid < bestRMSE) {
      bestRMSE = rmseValGrid
      bestRegParam= regParam
      bestModel = model
    }
    }

    val rmseValLRGrid = bestRMSE
    println("best rsme: " + rmseValLRGrid)

    println("two way inter: " + twoWayInteractions(LabeledPoint(0.0, Vectors.dense(Array(2.0, 3.0)))))
    val trainWithInteractionsDF = trainDF.map(x => twoWayInteractions(LabeledPoint(x(0).asInstanceOf[Double],
      Vectors.dense(x(1).asInstanceOf[org.apache.spark.ml.linalg.Vector].toArray))))
    val valWithInteractionsDF = valDF.map(x => twoWayInteractions(LabeledPoint(x(0).asInstanceOf[Double],
      Vectors.dense(x(1).asInstanceOf[org.apache.spark.ml.linalg.Vector].toArray))))
    val testWithInteractionsDF = testDF.map(x => twoWayInteractions(LabeledPoint(x(0).asInstanceOf[Double],
      Vectors.dense(x(1).asInstanceOf[org.apache.spark.ml.linalg.Vector].toArray))))


   // val numIters = 500
   // val reg = 1e-10
  //  val alpha = 0.2
  //  val useIntercept = true

    val linReg = new LinearRegression().setMaxIter(numIters)
      .setRegParam(reg)
      .setElasticNetParam(alpha)
      .setFitIntercept(useIntercept)
    val modelInteract = linReg.fit(trainWithInteractionsDF)
    val predsAndLabelsInteractDF = modelInteract.transform(valWithInteractionsDF)
    val rmseValInteract = evaluator.evaluate(predsAndLabelsInteractDF)
    println("interact rsme: " + rmseValInteract)

    val predsAndLabelsTestDF = modelInteract.transform(testWithInteractionsDF)
    val rmseTestInteract = evaluator.evaluate(predAndLabelsTest)
    println("interact rsme: " + rmseTestInteract)



    val linReg2 = new LinearRegression().setMaxIter(numIters)
      .setRegParam(reg)
      .setElasticNetParam(alpha)
      .setFitIntercept(useIntercept)
      .setFeaturesCol("polyFeatures")

    val polynomialExpansion = new PolynomialExpansion()
                                            .setDegree(2)
                                            .setInputCol("features")
                                            .setOutputCol("polyFeatures")



    val pipeline = new Pipeline().setStages(Array(polynomialExpansion,linReg2))
    val pipelineModel = pipeline.fit(trainDF)

    val predictionsDF = pipelineModel.transform(testDF)
    val rmseTestPipeline = evaluator.evaluate(predictionsDF)
    println("pipeline rmse: " + rmseTestPipeline)










        }

}
