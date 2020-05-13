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
    return df



def diff_between_sales():
    sales_total = df.resample("D")[['sales_total']].sum()
    sales_total['sales_differences'] = sales_total['sales_total'].diff()
    return sales_total.head()


def prep_ops():
    ops = acquire.get_opsd_data()
    ops['Date'] = pd.to_datetime(ops['Date'])
    ops = ops.sort_values('Date').set_index('Date')
    ops['month'] = ops.index.month
    ops['year'] = ops.index.year
    return ops