import duckdb, os
import boto3
session = boto3.Session()
credentials = session.get_credentials().get_frozen_credentials()
# s3 = boto3.resource(service_name='s3')
s3 = boto3.client('s3')


os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
os.environ['AWS_SESSION_TOKEN'] = credentials.token

duckdb.sql("INSTALL httpfs; LOAD httpfs;")

duckdb.sql(f"""
    SET s3_region='ap-southeast-2';  
    SET s3_access_key_id='{credentials.access_key}';
    SET s3_secret_access_key='{credentials.secret_key}';
    SET s3_session_token='{credentials.token}';
""")


