from sqlalchemy.engine import create_engine
from sqlalchemy import text
import subprocess as sp
import pandas as pd
from time import sleep
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import boto3
session = boto3.Session()
s3 = boto3.client('s3')

pool_size = 8
max_overflow = 20
pool_timeout = 90
trino_hive = create_engine("trino://ubuntu@trino.ciccada:8080/hive/solar_analytics", pool_size=pool_size, max_overflow=max_overflow, pool_timeout=pool_timeout)
trino_iceberg = create_engine("trino://ubuntu@trino.ciccada:8080/iceberg/solar_analytics_iceberg", pool_size=pool_size, max_overflow=max_overflow, pool_timeout=pool_timeout)
trino_bom = create_engine("trino://ubuntu@trino.ciccada:8080/iceberg/BOM_NCI", pool_size=pool_size, max_overflow=max_overflow, pool_timeout=pool_timeout)

def make_trino_engine(catalog):
    return create_engine(
        f"trino://ubuntu@trino.ciccada:8080/{catalog}",
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True
    )

def hive_sql(query: str) -> pd.DataFrame:
    with trino_hive.connect() as conn: 
        df = pd.read_sql(query, conn)
    return df

def iceberg_sql(query: str) -> pd.DataFrame:
    engine = make_trino_engine("iceberg/solar_analytics_iceberg")
    with engine.connect() as conn: 
        # conn.execute(text("SET SESSION query_max_memory_per_node = '45GB'"))
        conn.execute(text("SET SESSION task_concurrency = 1"))
        result = conn.execution_options(stream_results=True).execute(text(query))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        result.close()
        engine.dispose()
    return df

def iceberg_exec(query):
    with trino_iceberg.connect() as conn:
        # conn.execute(text("SET SESSION query_max_run_time = '60m'"))
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


def trino_parallel_batch(run_func, tasks, num_workers=1, batch_size=4):
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_func, task) for task in batch]
            batch_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        batch_results.append(result)
                except Exception as e:
                    print(f"Error in task: {e}")
        # ✅ Process/save after every 4 tasks
        if batch_results:
            # print(f"Saving batch {i // batch_size + 1}")
            results.extend(batch_results)
            sleep(10)  # Sleep after processing each batch to reduce load on Trino
            if (i // batch_size + 1) % 5 == 0:  # Sleep for a longer duration after every 5 batches
                print("Sleeping for 30 seconds to reduce load on Trino...")
                sleep(30)
    if results:
        print("Combining all batch results.")
        return pd.concat(results, ignore_index=True)

    return None

def trino_parallel(run_func, tasks, num_workers=1):
    df_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(run_func, task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                df_list.append(future.result())
            except Exception as e:
                task = future_to_task[future]
                print(f"Error in task {task}: {e}")
    df_list = [df for df in df_list if df is not None]
    if len(df_list) > 0:
        print("Combining results from all tasks.")
        return pd.concat(df_list, ignore_index=True)
    else:
        return None

region='ap-southeast-2'
service = 'trino-service'
worker_service = 'worker-trino-service'
big_worker_service = 'big-worker-trino-service'
cluster = 'my-ecs-cluster'

def ensure_trino_running(worker_desired_count=1, big_worker_desired_count=0):
    cmd = f"aws ecs describe-services --cluster {cluster} --services {service} --query services[0].desiredCount --output text"

    trino_count = sp.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    cmd = f"aws ecs describe-services --cluster {cluster} --services {worker_service} --query services[0].desiredCount --output text"

    worker_count = sp.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    cmd = f"aws ecs describe-services --cluster {cluster} --services {big_worker_service} --query services[0].desiredCount --output text"

    big_worker_count = sp.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    if trino_count == '0' or worker_count != str(worker_desired_count) or big_worker_count != str(big_worker_desired_count):
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
        sp.run(f"aws  ecs wait services-stable --cluster {cluster} --services {service} {worker_service} {big_worker_service}", shell=True, check=True)
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
