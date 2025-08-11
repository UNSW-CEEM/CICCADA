import boto3, os, sys

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python")
spark_home = "/home/ubuntu/spark-4.0.0-bin-hadoop3"
sys.path.insert(0, os.path.join(spark_home, "python"))
sys.path.insert(0, os.path.join(spark_home, "python", "lib", "py4j-0.10.9.9-src.zip"))

