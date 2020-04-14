import env
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'



def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))



def get_iris_data():
    return pd.read_sql('select * from measurements join species using (species_id)', get_connection('iris_db'))