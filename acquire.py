import pandas as pd
import numpy as np
import env
import os
from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_titanic_data():
    sql_query = 'SELECT * FROM passengers'
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    return df


def get_titanic_data():
    if os.path.isfile('titanic_df.csv'):
        df = pd.read_csv('titanic_df.csv', index_col=0)
    else:
        df = new_titanic_data()
        df.to_csv('titanic_df.csv')
    return df

def new_iris_data():
    sql_query = """
                SELECT
                    species_id,
                    species_name,
                    measurement_id,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
                """
                
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    return df

def get_iris_data():
    if os.path.isfile('iris_df.csv'):
        df = pd.read_csv('iris_df.csv', index_col =0)
    else:
        df = new_iris_data()
        df.to_csv('iris_df.csv')
    return df

