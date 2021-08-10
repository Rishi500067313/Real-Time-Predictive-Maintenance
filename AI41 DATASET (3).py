# Databricks notebook source
dataset = spark.read.load("dbfs:/FileStore/tables/ai4i2020___ai4i2020__6_.csv", format = "csv", header= "true", inferSchema = "true")

from pyspark.sql.functions import col,count,when
dataset.select([count(when(col(c).isNull(), c)).alias(c) for c in dataset.columns]).show()

trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=42)
print(trainDF.cache().count()) # Cache because accessing training data multiple times
print(testDF.count())

display(trainDF)

display(trainDF.select("Air_temperature").summary())

display(trainDF
        .groupBy("Type")
        .count()
        .sort("count", ascending=False))

from pyspark.ml.feature import StringIndexer, OneHotEncoder
 
categoricalCols = ["Product_ID", "Type"]
 
# The following two lines are estimators. They return functions that we will later apply to transform the dataset.
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols]).setHandleInvalid("keep")
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols]) 
 
# The label column ("Machine failure") is also a string value - it has two possible values, "PASS" and "FAIL". 
# Convert it to a numeric value using StringIndexer.
labelToIndex = StringIndexer(inputCol="Machine_failure", outputCol="label")


stringIndexerModel = stringIndexer.fit(trainDF)
display(stringIndexerModel.transform(trainDF))

from pyspark.ml.feature import VectorAssembler
 
# This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
numericCols = ["UDI", "Air_temperature", "Process_temperature", "Rotational_speed", "Torque","TWF",
              "HDF","PWF","OSF","RNF"]
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")


from pyspark.ml.classification import LogisticRegression
 
lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=1.0)


from pyspark.ml import Pipeline
 
# Define the pipeline based on the stages created in previous steps.
pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex , vecAssembler, lr])
 
# Define the pipeline model.
pipelineModel = pipeline.fit(trainDF)
 
# Apply the pipeline model to the test dataset.
predDF = pipelineModel.transform(testDF)


display(predDF.select("features", "label", "prediction", "probability"))


display(pipelineModel.stages[-1], predDF.drop("prediction", "rawPrediction", "probability"), "ROC")


from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
 
bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predDF)}")
 
mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(predDF)}")


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
 
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())


# Create a 3-fold CrossValidator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=bcEvaluator, numFolds=3, parallelism = 4)
 
# Run cross validations. This step takes a few minutes and returns the best model found from the cross validation.
cvModel = cv.fit(trainDF)


# Use the model identified by the cross-validation to make predictions on the test dataset
cvPredDF = cvModel.transform(testDF)
 
# Evaluate the model's performance based on area under the ROC curve and accuracy 
print(f"Area under ROC curve: {bcEvaluator.evaluate(cvPredDF)}")
print(f"Accuracy: {mcEvaluator.evaluate(cvPredDF)}")


cvPredDF.createOrReplaceTempView("finalPredictions")


# After reading data from eventhub 


rdd4 = spark.sql("""select 
          substring(Data,1,charindex(' ',data)-1) as UDI, 
          substring(data,4,charindex(' ',data)+4) as Product_ID, 
          substring(data,11,charindex(' ',data)-1) as Type, 
          substring(data,14,charindex(' ',data)+0) as Air_temperature,
          substring(data,17,charindex(' ',data)+1) as Process_temperature,
          substring(data,21,charindex(' ',data)+2) as Rotational_speed,
          substring(data,26,charindex(' ',data)+0) as Torque,
          substring(data,29,charindex(' ',data)-1) as TWF,
          substring(data,31,charindex(' ',data)-1) as HDF,
          substring(data,33,charindex(' ',data)-1) as PWF,
          substring(data,35,charindex(' ',data)-1) as OSF,
          substring(data,37,charindex(' ',data)-1) as RNF from i2""").rdd


df = rdd4.toDF()
from pyspark.sql.types import DecimalType, IntegerType
df1 = df.withColumn("UDI", df["UDI"].cast(DecimalType())).withColumn("Air_temperature", df["Air_temperature"].cast(DecimalType())).withColumn("Process_temperature", df["Process_temperature"].cast(DecimalType())).withColumn("Rotational_speed", df["Rotational_speed"].cast(DecimalType())).withColumn("Torque", df["Torque"].cast(DecimalType())).withColumn("TWF", df["TWF"].cast(IntegerType())).withColumn("HDF", df["HDF"].cast(IntegerType())).withColumn("PWF", df["PWF"].cast(IntegerType())).withColumn("OSF", df["OSF"].cast(IntegerType())).withColumn("RNF", df["RNF"].cast(IntegerType()))


df1.show()


test = cvModel.transform(df1)
test.createOrReplaceTempView("finalPredictions")


spark.sql("describe extended finalPredictions").show()


spark.sql("select UDI, probability, prediction from finalPredictions").show(200)



