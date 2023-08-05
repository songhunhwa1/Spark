from pyspark.sql.types import *
from pyspark.sql import Row

# create a rdd 
rdd = sc.textFile('Desktop/pyspark_ml/realEstate.csv')
rdd.take(5)

# show the first 5 rows from the rdd
rdd = rdd.map(lambda line: line.split(","))
rdd.take(3)

# get rid of the first row
header = rdd.first()
rdd = rdd.filter(lambda line:line != header)

# select cols that are needed
df = rdd.map(lambda line: Row(street = line[0], city = line[1], zip=line[2], beds=line[4], baths=line[5], sqft=line[6], price=line[9])).toDF()

# to pandas df
df.toPandas().head()

# Regression with MLlib
import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *

df = df.select('price', 'baths', 'beds', 'sqft')
df = df.filter("baths != 0").filter("beds != 0").filter("sqft != 0")

# LabeledPoints: MLlib requires that our features be expressed with LabeledPoints. 
# The required format for a labeled point is a tuple of the response value and a vector of predictors. 
df_rdd = df.rdd
temp = df_rdd.map(lambda line: LabeledPoint(line[0], [line[1:]]))

# import modules
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler

# features scale
features = df_rdd.map(lambda row: row[1:])
standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)

# merge with target var
# RDDs can be put together with 'zip'
lab = df_rdd.map(lambda row:row[0])
transformData = lab.zip(features_transform)

transformData = transformData.map(lambda row: LabeledPoint(row[0], [row[1]]))

# split the dataset
train, test = transformData.randomSplit([.8, .2], seed=1234)

# Stochastic Gradient Descent (SGD)
from pyspark.mllib.regression import LinearRegressionWithSGD

# set the interations and learning rate
linearModel = LinearRegressionWithSGD.train(train, 1000, .2)

# coefficients and intercepts
linearModel.weights

linearModel.predict([2.5, 3.4, 1.5])

# metrics
from pyspark.mllib.evaluation import RegressionMetrics

prediObserRDDin = train.map(lambda row: (float(linearModel.predict(row.features[0])), row.label))
metrics = RegressionMetrics(prediObserRDDin)
metrics.r2

prediObserRDDout = test.map(lambda row: (float(linearModel.predict(row.features[0])), row.label))
metrics = RegressionMetrics(prediObserRDDout)
metrics.r2
metrics.rootMeanSquaredError
metrics.meanSquaredError
metrics.explainedVariance
