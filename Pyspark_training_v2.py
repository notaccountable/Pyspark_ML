from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
import pyspark
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

conf = pyspark.SparkConf().setAppName('RF_pyspark').setMaster('local').setSparkHome('\SPARK')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

#spark.conf.set("spark.executor.cores", "4")

df = spark.read.csv("TrainingDataset_clean.csv", header=True, inferSchema=True)
for i in df:
    print ("These are your dataframes \n"+str(i))

feature_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "pH", "sulphates", "alcohol", "quality" ]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=100)
model = rf.fit(train_data)

predictions = model.transform(test_data)
#print('these are your predictions\n')
#print(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
accuracy = evaluator.evaluate(predictions)
print(f"F1 score: {accuracy}")

'''paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [50, 100, 150])
             .addGrid(rf.maxDepth, [5, 10, 15])
             .build())
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
'''

s3_path = 's3://cs643-s3-trained-model-bucket/saved-model/RF_model.model'
#spark.conf.set("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
model.save('RF_model.model')
print("this is your model \n" )
print(model)
#model.write.csv('RF_model.model')
#save_model_to_s3(spark, model, s3_path)

