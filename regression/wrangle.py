import pandas as pd
import numpy as np
from env import get_db_url

def get_data_from_mysql():
    query = """
    SELECT customer_id, monthly_charges, tenure, total_charges FROM customers
    JOIN contract_types USING (contract_type_id)
    WHERE contract_type_id = 3;
    """
    df = pd.read_sql(query, get_db_url("telco_churn"))
    return df


def wrangle_telco():
    df = get_data_from_mysql()
    df.total_charges = df.total_charges.str.strip().replace('', np.nan).astype(float)
    df = df.dropna() 
    return df
   