import env
import pandas as pd
import numpy as np

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow_data():
    sql = """
    select * from properties_2017
    join predictions_2017 using(parcelid)
    left join airconditioningtype using(airconditioningtypeid)
    left join architecturalstyletype using(architecturalstyletypeid)
    left join buildingclasstype using(buildingclasstypeid)
    left join heatingorsystemtype using(heatingorsystemtypeid)
    left join propertylandusetype using(propertylandusetypeid)
    left join storytype using(storytypeid)
    left join typeconstructiontype using(typeconstructiontypeid)
    where latitude IS NOT NULL
    and longitude IS NOT NULL
    """
    url = get_connection("zillow")
    return pd.read_sql(sql, url)