import subprocess as sp
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

import boto3
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import create_engine

session = boto3.Session()
s3 = boto3.client("s3")

pool_size = 8
max_overflow = 20
pool_timeout = 90
trino_hive = create_engine(
    "trino://ubuntu@trino.ciccada:8080/hive/solar_analytics",
    pool_size=pool_size,
    max_overflow=max_overflow,
    pool_timeout=pool_timeout,
)
trino_iceberg = create_engine(
    "trino://ubuntu@trino.ciccada:8080/iceberg/solar_analytics_iceberg",
    pool_size=pool_size,
    max_overflow=max_overflow,
    pool_timeout=pool_timeout,
)
trino_bom = create_engine(
    "trino://ubuntu@trino.ciccada:8080/iceberg/BOM_NCI",
    pool_size=pool_size,
    max_overflow=max_overflow,
    pool_timeout=pool_timeout,
)


def hive_sql(query: str) -> pd.DataFrame:
    with trino_hive.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def iceberg_sql(query: str) -> pd.DataFrame:
    with trino_iceberg.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def iceberg_exec(query):
    with trino_iceberg.connect() as conn:
        conn.execute(text(query))
        print("Executed")


def hive_exec(query):
    with trino_hive.connect() as conn:
        conn.execute(text(query))
        print("Executed")


def bom_exec(query):
    with trino_bom.connect() as conn:
        conn.execute(text(query))
        print("Executed")


def trino_parallel(run_func, tasks, num_workers=1):
    df_list = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(run_func, task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                df_list.append(future.result())
            except Exception as e:
                task = future_to_task[future]
                print(f"Error in task {task}: {e}")
    if len(df_list) > 0 and df_list[0] is not None:
        return pd.concat(df_list)
    else:
        return None


region = "ap-southeast-2"
service = "trino-service"
worker_service = "worker-trino-service"
big_worker_service = "big-worker-trino-service"
cluster = "my-ecs-cluster"


def ensure_trino_running(worker_desired_count=1, big_worker_desired_count=0):
    cmd = f"aws ecs describe-services --cluster {cluster} --services {service} --query services[0].desiredCount --output text"

    trino_count = sp.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    cmd = f"aws ecs describe-services --cluster {cluster} --services {worker_service} --query services[0].desiredCount --output text"

    worker_count = sp.run(
        cmd, shell=True, capture_output=True, text=True
    ).stdout.strip()

    cmd = f"aws ecs describe-services --cluster {cluster} --services {big_worker_service} --query services[0].desiredCount --output text"

    big_worker_count = sp.run(
        cmd, shell=True, capture_output=True, text=True
    ).stdout.strip()

    if (
        trino_count == "0"
        or worker_count != str(worker_desired_count)
        or big_worker_count != str(big_worker_desired_count)
    ):
        print("Trino service is not running. Starting the service...")
        start_cmd = f"aws ecs update-service \
            --cluster {cluster} \
            --service {service} \
            --desired-count 1"
        sp.run(start_cmd, shell=True, capture_output=True, text=True).stdout.strip()

        sleep(5)

        start_cmd = f"aws ecs update-service \
            --cluster {cluster} \
            --service {worker_service} \
            --desired-count {worker_desired_count}"
        sp.run(start_cmd, shell=True, capture_output=True, text=True).stdout.strip()

        start_cmd = f"aws ecs update-service \
            --cluster {cluster} \
            --service {big_worker_service} \
            --desired-count {big_worker_desired_count}"
        sp.run(start_cmd, shell=True, capture_output=True, text=True).stdout.strip()

        print("Trino service triggered.")
        sp.run(
            f"aws  ecs wait services-stable --cluster {cluster} --services {service} {worker_service} {big_worker_service}",
            shell=True,
            check=True,
        )
        print(f"Service {service} is now stable.")
    else:
        print("Trino service is already running.")


def stop_trino():
    cmd = f"aws ecs update-service \
        --cluster {cluster} \
        --service {service} \
        --desired-count 0"
    sp.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    sleep(1)

    cmd = f"aws ecs update-service \
        --cluster {cluster} \
        --service {worker_service} \
        --desired-count 0"
    sp.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    cmd = f"aws ecs update-service \
        --cluster {cluster} \
        --service {big_worker_service} \
        --desired-count 0"
    sp.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    print("Trino service stopping triggered.")
