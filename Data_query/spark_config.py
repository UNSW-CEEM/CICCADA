import os
import sys

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python")
spark_home = "/home/ubuntu/spark-4.0.0-bin-hadoop3"
sys.path.insert(0, os.path.join(spark_home, "python"))
sys.path.insert(0, os.path.join(spark_home, "python", "lib", "py4j-0.10.9.9-src.zip"))

from pyspark.sql import SparkSession
from pyspark.sql.functions import abs as spark_abs
from pyspark.sql.functions import (
    avg,
    col,
    concat,
    date_format,
    dayofmonth,
    dense_rank,
    explode,
    expr,
    from_json,
    greatest,
    hour,
    lag,
    least,
    lit,
    month,
    row_number,
    to_date,
    to_timestamp,
    udf,
    unix_timestamp,
    when,
    year,
)
from pyspark.sql.functions import count as spark_count
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window

spark = (
    SparkSession.builder.appName("S3Access")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider",
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.ap-southeast-2.amazonaws.com")
    .config("spark.sql.warehouse.dir", "s3a://project-ciccada/spark-warehouse/")
    .config("spark.hadoop.hive.metastore.uris", "thrift://localhost:9083")
    .config("spark.sql.catalogImplementation", "hive")
    .config("spark.local.dir", "/mnt/spark-temp")
    .config("spark.driver.memory", "40g")
    .config("spark.sql.hive.metastore.jars", "/home/ubuntu/hive-4.0.1/lib/*")
    .config("spark.sql.hive.metastore.version", "4.0.1")
    .enableHiveSupport()
    .getOrCreate()
)

# .config("spark.local.dir", "/home/ubuntu/tmp") \
# .config("spark.local.dir", "/mnt/spark-temp") \
