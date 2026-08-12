import json
import os
import urllib.parse

import pandas as pd
import psycopg2
import yaml
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder


def get_darth_data(config, sql_query):
    with SSHTunnelForwarder(
        ssh_address_or_host=(config["hostserver"], 22),
        ssh_username=config["ssh_username"],
        ssh_pkey=config["ssh_private_key_path"],
        ssh_private_key_password=config["ssh_private_key_password"],
        host_pkey_directories=[],
        remote_bind_address=(config["remote_host"], config["remote_port"]),
    ) as tunnel:
        tunnel.start()

        local_port = str(tunnel.local_bind_port)

        # Needed to handle special characters in password i.e. "@"
        darth_password = urllib.parse.quote_plus(config["darth_password"])

        engine_str = (
            "postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}".format(
                user=config["darth_username"],
                password=darth_password,
                host="localhost",
                port=local_port,
                db=config["databasename"],
            )
        )

        engine = create_engine(engine_str)
        con = engine.raw_connection()

        try:
            df = pd.read_sql_query(sql_query, con=con)

        except (Exception, psycopg2.DatabaseError) as err:
            print(err)

        finally:
            con.close()
            engine.dispose()

    return df
