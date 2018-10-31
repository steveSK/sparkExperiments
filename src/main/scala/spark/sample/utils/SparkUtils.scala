package spark.sample.utils

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{Dataset, Row, SparkSession}

/**
  * Created by stefan on 11/9/16.
  */
object SparkUtils {

  def  parseRawDF(rawDF: Dataset[Row], labelIndex: Int, sparkSession: SparkSession): Dataset[Row] = {
    import sparkSession.implicits._


    val convertedRawDF = rawDF.rdd.map(x => x.getString(0).split(", ")).toDF()
    val getLabel: Seq[String] => Double = x => x(labelIndex) match {
      case "<=50K" => 0.0
      case ">=50K" => 1.0
    }
    val getLabelUDF = udf(getLabel)

    val getFeatures: Seq[String] => Seq[String] = x => x.slice(1, x.size)
    val getFeaturesUDF = udf(getFeatures)

    return convertedRawDF.withColumn("label", getLabelUDF($"value"))
      .withColumn("features", getFeaturesUDF($"value")).drop("value")

  }

  def loadDataFromCSV(filepath: String, sparkSession: SparkSession): Dataset[Row] = {
    sparkSession.read.option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .format("csv").load(filepath)
  }

  def transformRawDFToLabeledPointRDD(df: Dataset[Row],continousColumns : List[String], toLabel: String => Double, sparkSession: SparkSession): RDD[LabeledPoint] = {
    import sparkSession.implicits._

    val indexTransformers: List[StringIndexer] = continousColumns.map(
      cname => new StringIndexer()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_index")
    )

    val indexModels = indexTransformers.map(x => x.fit(df.toDF()))

    var indexedDF = df
    for (indexer <- indexModels) {
      indexedDF = indexer.transform(indexedDF)
    }
    val indexColumns = indexedDF.columns.filter(x => x contains "index")
    //   indexColumns.foreach(println)


    val oneHotEncoders: Array[OneHotEncoder] = indexColumns.map(
      cname => new OneHotEncoder()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_vec")
    )

    var oheDF = indexedDF
    for (ohe <- oneHotEncoders) {
      oheDF = ohe.transform(oheDF)
    }

    val columnsToDrop = continousColumns ++ indexColumns
    val oheTrainDF = oheDF.drop(columnsToDrop: _*)
    //oheTrainDF.show()

    val columnsToAssemble = oheTrainDF.columns.filter(x => x != "label");

    val assembler = new VectorAssembler()
      .setInputCols(columnsToAssemble)
      .setOutputCol("features")

    val getLabelUDF = udf(toLabel)


    val transformedTrainDf = assembler.transform(oheTrainDF).select(getLabelUDF($"label"), $"features")
    return transformedTrainDf.rdd.map(x => LabeledPoint(x.getDouble(0), org.apache.spark.mllib.linalg.Vectors.dense(x.getAs[org.apache.spark.ml.linalg.Vector](1).toArray)))
  }

}
