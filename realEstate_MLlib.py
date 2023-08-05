# read the csv file
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np

# read the dataset
df = sqlContext.read.format('com.databricks.spark.csv')\
					.options(header='true', inferSchema='true')\
					.load('/Users/woowahan/Documents/Pyspark_MLlib/realEstate.csv')

# preprocessing
df1 = df.select(df.price.cast('integer'), df.beds.cast('integer'), df.baths.cast('integer'), df.sq__ft.cast('integer'))\
		.withColumnRenamed('sq__ft', 'sqft')\
		.filter("baths != 0 and beds != 0 and sqft != 0")

# Scale & Modeling
assembler = VectorAssembler(inputCols = ['beds', 'baths', 'sqft'], outputCol = 'features')
#df2 = assembler.transform(df1)

standardizer = StandardScaler(withMean=True, withStd=True, inputCol='features', outputCol='std_features')
#scale_model = standardizer.fit(df2)
#df3 = scale_model.transform(df2)

#df4 = df3.select(['price', 'std_features'])

lr_model = LinearRegression(featuresCol='features', labelCol='price')

# Pipeline
pipeline = Pipeline(stages=[assembler, lr_model])

# split the dataset
train, test = df1.randomSplit([0.8, 0.2], seed=123)

# train the model
model = pipeline.fit(train)

# predict
prediction = model.transform(test)

# metrics
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# for no pipeline
# summary = model.summary
# print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
# print("T Values: " + str(summary.tValues))
# print("P Values: " + str(summary.pValues))
# print("Dispersion: " + str(summary.dispersion))
# print("Null Deviance: " + str(summary.nullDeviance))
# print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
# print("Deviance: " + str(summary.deviance))
# print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
# print("AIC: " + str(summary.aic))
# print("Deviance Residuals: ")
# summary.residuals().show()
