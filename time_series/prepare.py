import pandas as pd
import acquire
import datetime


def prep_sales_data():
    df = acquire.get_all_data(use_cache=True)
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df.sale_amount.plot().set_title('The distribution of sale amount over time')
    df.item_price.plot().set_title('The distribution of item price over time')
    df = df.sort_values('sale_date').set_index('sale_date')
    df["month"] = df.index.month_name()
    df["day_of_week"] = df.index.day_name()
    df['sales_total'] = df.sale_amount * df.item_price
    return df.head()



def diff_between_sales():
    sales_total = df.resample("D")[['sales_total']].sum()
    sales_total['sales_differences'] = sales_sum['sales_total'].diff()
    return sales_total.head()