import os, sys

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python")
spark_home = "/home/ubuntu/spark-4.0.0-bin-hadoop3"
sys.path.insert(0, os.path.join(spark_home, "python"))
sys.path.insert(0, os.path.join(spark_home, "python", "lib", "py4j-0.10.9.9-src.zip"))

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, explode, from_json, udf, to_timestamp, to_date, when, dayofmonth, month, year, hour, greatest, lit, \
    sum as spark_sum, lag, min as spark_min, count as spark_count, max as spark_max, expr, avg, least, concat, row_number, dense_rank, \
        date_format, unix_timestamp, abs as spark_abs
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, DoubleType, IntegerType, BooleanType, TimestampType, LongType
spark = SparkSession.builder \
    .appName("S3Access") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider") \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.ap-southeast-2.amazonaws.com") \
    .config("spark.sql.warehouse.dir", "s3a://project-ciccada/spark-warehouse/") \
    .config("spark.hadoop.hive.metastore.uris", "thrift://localhost:9083") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("spark.local.dir", "/mnt/spark-temp") \
    .config("spark.driver.memory", "40g") \
    .config("spark.sql.hive.metastore.jars", "/home/ubuntu/hive-4.0.1/lib/*") \
    .config("spark.sql.hive.metastore.version", "4.0.1") \
    .enableHiveSupport() \
    .getOrCreate()

    # .config("spark.local.dir", "/home/ubuntu/tmp") \
    # .config("spark.local.dir", "/mnt/spark-temp") \
