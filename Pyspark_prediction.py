from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
import pyspark

from pyspark.sql import SparkSession

#spark = SparkSession.builder.getOrCreate()


conf = pyspark.SparkConf().setAppName('Pyspark_prediction')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)


#model_path = 's3://cs643-s3-trained-model-bucket/saved-model/RF_model.model'
model_path = 'RF_model.model'
model = PipelineModel.load(model_path)

#data_path = 's3://cs643-s3-trained-model-bucket/saved-model/RF_model.model/data/part-00000-6953ac0e-efbd-4ab8-b89b-225c21a15b0e-c000.snappy.parquet'
data_path = 'RF_model.model/data/part-00000-6953ac0e-efbd-4ab8-b89b-225c21a15b0e-c000.snappy.parquet'
data = spark.read.parquet(data_path, header=True, inferSchema=True)

#loaded_model = PipelineModel.load('s3://cs643-s3-trained-model-bucket/saved-model/RF_model.model')

print('model loaded properly \n')

predictions = model.transform(data)
print('these are your predictions\n')
print(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
accuracy = evaluator.evaluate(predictions)
print(f"F1 score: {accuracy}")