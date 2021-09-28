// Databricks notebook source
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.eventhubs._
import com.microsoft.azure.eventhubs._

val namespaceName = "azuretrial"
val eventHubName = "final"
val sasKeyName = "root"
val sasKey = "<Primary Key>"
val connStr = new com.microsoft.azure.eventhubs.ConnectionStringBuilder()
            .setNamespaceName(namespaceName)
            .setEventHubName(eventHubName)
            .setSasKeyName(sasKeyName)
            .setSasKey(sasKey)

val customEventhubParameters = EventHubsConf(connStr.toString()).setMaxEventsPerTrigger(50)

val incomingStream = spark.readStream.format("eventhubs").options(customEventhubParameters.toMap).load()

val messages =
  incomingStream
  .withColumn("Offset", $"offset".cast(LongType))
  .withColumn("Time (readable)", $"enqueuedTime".cast(TimestampType))
  .withColumn("Timestamp", $"enqueuedTime".cast(LongType))
  .withColumn("Data", $"body".cast(StringType))
  .select("Data")

//messages.writeStream.outputMode("append").format("console").option("truncate", false).start().awaitTermination()

messages.writeStream.format("delta").outputMode("append").option("checkpointLocation","/data/events/_checkpoints/data_file_1").table("i2")

// COMMAND ----------

val messages = spark.readStream.table("i2").select("Data")

// COMMAND ----------

val query = messages.writeStream.outputMode("append").format("console").start()


