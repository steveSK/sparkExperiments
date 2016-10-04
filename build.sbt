import AssemblyKeys._

assemblySettings

name := "spark_test_project"

version := "1.0"

scalaVersion := "2.10.5"

resolvers += Resolver.mavenLocal

resolvers += "justwrote" at "http://repo.justwrote.it/releases/"

libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "1.6.2"

libraryDependencies += "org.apache.spark" % "spark-sql_2.10" % "1.6.2"

libraryDependencies += "it.justwrote" %% "scala-faker" % "0.3"


    
