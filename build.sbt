
name := "spark_experiments"

version := "1.0"

scalaVersion := "2.11.7"

resolvers += Resolver.mavenLocal

resolvers += "justwrote" at "http://repo.justwrote.it/releases/"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.1.0"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.1.0"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.1.0"

libraryDependencies += "it.justwrote" %% "scala-faker" % "0.3"


    
