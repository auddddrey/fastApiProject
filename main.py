import os

import pandas as pd
from fastapi import FastAPI
from numpy import double
from pydantic import BaseModel
from pyspark import SparkConf, SparkContext
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressionModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

#  Initialize Spark
conf = SparkConf().setAppName("myApp")
sc = SparkContext(conf=conf)
# print(sc._jsc.sc().isStopped())

#  Initialize SparkSession
spark = SparkSession.builder.appName("myApp").getOrCreate()

#  Initialize FastAPI
# jvm = sc._jvm
app = FastAPI()

mPath = os.path.join(os.getcwd(), "models", "model")
print(mPath)

# Load the model
model = GBTRegressionModel.load(mPath)

# Load the pipeline model
model_path2 = os.path.join(os.getcwd(), "models", "pipelinemodel");
pipeline_model = PipelineModel.load(model_path2)

# Define the schema for the input data
schema = StructType([
    StructField("color", StringType(), True),
    StructField("cut", StringType(), True),
    StructField("clarity", StringType(), True),
    StructField("carat", DoubleType(), True),
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True),
    StructField("z", DoubleType(), True),
    StructField("depth", DoubleType(), True),
    StructField("table", DoubleType(), True),
])


# Define the input data class Pydantic model
class Item(BaseModel):
    color: str  # D, E, F, G, H, I, J
    cut: str  # Fair, Good, Very Good, Premium, Ideal
    clarity: str  # I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF
    carat: double  # 0.2 - 5.01
    x: double  # 0 - 10.74
    y: double  # 0 - 58.9
    z: double  # 0 - 31.8
    depth: double  # 43 - 79
    table: double  # 43 - 95


# Define the input data columns
INPUT_COLS = ["color", "cut", "clarity", "carat", "x", "y", "z", "depth", "table"]


# Define the data transformation function
def transform_data(sdf):
    # Prepare input data for model
    sdf = sdf.select(INPUT_COLS)

    # Load the input data, convert the categorical columns to numeric
    categorical_columns = ['cut', 'color', 'clarity']
    sdf = pipeline_model.transform(sdf)
    sdf = sdf.drop(*categorical_columns)

    feature = VectorAssembler(inputCols=sdf.columns, outputCol="features")
    feature_vector = feature.transform(sdf)
    return feature_vector


# Create an API endpoint for getting the predicted price of diamond
@app.post('/logprice')
async def logprice_endpoint(item: Item):
    try:
        df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
        sdf = spark.createDataFrame(df, schema=schema)

        # If necessary, apply any transformations to the selected columns here
        sdf = transform_data(sdf)
        yhat = model.transform(sdf)

        return {"logprice_predicted": yhat.select("prediction").first()[0]}
    except Exception as e:
        return {"error": str(e)}

#   Test endpoint
@app.get('/')
async def init_endpoint():
    try:
        return {"hello": "world"}
    except Exception as e:
        return {"error": str(e)}
