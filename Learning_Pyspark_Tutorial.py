# create a rdd
st = sc.parallelize([
(123, 'KT', 19, 'brown'),
(332, 'ER', 19, 'brown'),
(43, 'SD', 33, 'red'),
(12, 'TI', 34, 'green'),
])

## Read Dataset ------------
# read the csv with library
df = sqlContext.read.format('com.databricks.spark.csv')\
					.options(header='true', inferSchema='true')\
					.load('/Users/songhunhwa/Documents/Python/Pyspark_MLlib/data/Default.csv')\
					.drop("_c0")

# read a csv as in text
df_rdd = sc.textFile('/Users/songhunhwa/Documents/Pyspark_MLlib/data/Default.csv')
header = df_rdd.first()
df_rdd = df_rdd.filter(lambda row: row != header) \
			   .map(lambda row: [e for e in row.split(',')])


# apply the schema
from pyspark.sql.types import *

schema = StructType([
StructField("id", LongType(), True),
StructField("name", StringType(), True),
StructField("age", LongType(), True),
StructField("color", StringType(), True)
])

# duplicates
df.count()
df.select("default").distinct().show()
df.select("default").distinct().count()
df.select("default").dropDuplicates().show()
df_tt = df.dropDuplicates(subset=[c for c in df.columns if c != 'student'])

from pyspark.sql.functions import *
df.agg(count("student").alias("count_stu"), countDistinct("student").alias("unique_stu")).show()

# generate unique ids
df_newid = df.withColumn("new_id", monotonically_increasing_id()) # very useful !!!
df_newid.where('new_id == 1').show()

# missing values
df.rdd.map(
	lambda row: (row[1], sum([c == None for c in row]))
	).collect()

means = df.agg(*[fn.mean(c).alias(c)
					for c in df.columns]).toPandas().to_dict('records')[0]

# outliers
for col in cols:
	quantiles = df.approxQuantile(
		col, [0.25, 0.75], 0.05)	
	IQR = quantiles[1] - quantiles[0]
	bounds[col] = [
		quantiles[0] - 1.5 * IQR,
		quantiles[1] + 1.5 * IQR]

outliers = df.select([
	(
		(df[c] < bounds[c][0]) | (df[c] > bounds[c][1])
		).alias(c + '_o') for c in cols
	])

# corr
df.corr('crim', 'zn')

# histogram
hist = df.select('crim').rdd.flatMap(lambda row: row).histogram(20)
pd.DataFrame(zip(list(hist)[0], list(hist)[1]), columns=['bin','frequency']).set_index('bin')

# descriptive
import pyspark.mllib.stat as st
import numpy as np

n_rdd = df.rdd.map(lambda row: [e for e in row])
mlib_stats = st.Statistics.colStats(n_rdd)

mlib_stats.mean()
mlib_stats.variance()

# ml libraray tutorail
df = sqlContext.read.format('com.databricks.spark.csv')\
					.options(header='true', inferSchema='true')\
					.load('/Users/songhunhwa/Documents/Python/Pyspark_MLlib/data/births_transformed.csv')

# create transformers
from pyspark.ml.feature import *

births = df.withColumn("BIRTH_PLACE_INT", df['BIRTH_PLACE'].cast('integer'))
encoder = OneHotEncoder(inputCol = 'BIRTH_PLACE_INT', outputCol = 'BIRTH_PLACE_VEC')
featureCreator = VectorAssembler(inputCols = births.columns[1:], outputCol='features')

from pyspark.ml.classification import LogisticRegression
logistic = LogisticRegression(maxIter=10, regParam=0.01, labelCol='INFANT_ALIVE_AT_REPORT')

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[encoder, featureCreator, logistic])

# split
(train, test) = births.randomSplit([0.7, 0.3], seed=321)

# fit
model = pipeline.fit(train)
test_model = model.transform(test)

test_model.take(1)

# cross val
from pyspark.ml.evaluation import *

evaluator = BinaryClassificationEvaluator(rawPredictionCol='probability', labelCol='INFANT_ALIVE_AT_REPORT')

evaluator.evaluate(test_model)

# Save
pipelinePath = './infant_oneHotEncoder_Logistic_Pipeline'
pipeline.write().overwrite().save(pipelinePath)







