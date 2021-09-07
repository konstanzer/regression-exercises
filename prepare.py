from acquire import get_telco_data
import pandas as pd
import numpy as np
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point, Polygon

from sklearn.model_selection import train_test_split


def wrangle_zillow(df, test_size=.15, k=1.5):
    
    df=df.rename(columns={'calculatedfinishedsquarefeet':'finishedsqft',
                          'lotsizesquarefeet':'lotsqft',
                          'structuretaxvaluedollarcnt':'structuretaxvalue',
                          'yearbuilt':'year', 'taxvaluedollarcnt':'taxvalue',
                          'landtaxvaluedollarcnt':'landtaxvalue'})
    
    for col in df.columns:
        try:
            df[col].fillna(np.nanmedian(df[col]), inplace=True)
        except:
            continue

    df['bedroomcnt'].replace(0, np.median(df.bedroomcnt), inplace=True)
    df['bathroomcnt'].replace(0, np.median(df.bathroomcnt), inplace=True)
    df['roomcnt'].replace(0, 6, inplace=True)
    
    #if MVP columns are missing, drop it
    df=df.dropna(axis=0, subset=["taxvalue", "finishedsqft"])

    #engineered to show what proportion of lot is finished space 
    df['livingarearatio'] = df.finishedsqft/df.lotsqft 
    
    #engineered to show what proportion building is of total value
    df['buildinglandvalueratio'] = df.structuretaxvalue/df.landtaxvalue 
    df['taxrate'] = df.taxamount/df.taxvalue 

    df.latitude, df.longitude = df.latitude/1e6, df.longitude/1e6
    
    #handling outliers assuming no distribution
    cols = ['taxvalue', 'taxamount', 'finishedsqft', 'taxrate', 'structuretaxvalue']
    df = iqr_method(df, k, cols)
        
    int_cols = ['fips', 'bedroomcnt', 'bathroomcnt', 'taxvalue', 'taxamount', 'roomcnt',
                'finishedsqft', 'year', 'structuretaxvalue', 'landtaxvalue', 'lotsqft', 'regionidzip']
    df[int_cols] = df[int_cols].astype('int')
    
    #why not?
    zipzies = pd.get_dummies(df.regionidzip, drop_first=True)
    counties = pd.get_dummies(df.fips, drop_first=True)
    df = pd.concat([df, zipzies, counties], axis=1)
    
    counties = {6111:"Ventura", 6037:"Los Angeles", 6059:"Orange"}
    df['county'] = df.fips.map(counties)
    
    df=df.drop(columns=['id.1', 'id', 'parcelid', 'transactiondate', 'taxdelinquencyflag', 'fips',
                       'taxdelinquencyyear', 'propertyzoningdesc', 'propertycountylandusecode',
                        'buildingclasstypeid'])
    
    y = df.pop('taxvalue')
    
    return split_data(df, y, test_size)


def iqr_method(df, k, cols):
    
    #drop row if column outside fences
    #k is usu. between 1.5 and 3
    for col in cols:
        
        q1, q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upperbound, lowerbound = q3 + k*iqr, q1 - k*iqr
        df = df[(df[col] > lowerbound) & (df[col] < upperbound)]
        
    return df


def z_score_method(df, cols):
    
    #drop row if column has a z-score over 3, must be normal
    for col in cols:
        df = df[(np.abs(stats.zscore(df[col])) < 3)]
        
    return df
    
    
def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=36)
    test_size2 = test_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size2, random_state=36)
    print("X_train, X_test, X_val, y_train, y_test, y_val")
    print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def geo_df(df, y, file):
    map_ = gpd.read_file(file)
    geom = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    df['taxvalue'] = y
    geo_df = gpd.GeoDataFrame(df, crs = {'init':'epsg:4326'}, geometry=geom)
    
    return geo_df, map_


def telco_model_prep():
    
    telco = get_telco_data()
    cid = telco.pop("customer_id")
    
    #before making total charges a float, eliminate this annoying space
    telco = telco.replace(" ", 0)
    
    telco = telco.replace("No internet service", "No")
    telco = telco.replace("No phone service", "No")
    
    telco = telco.astype({'total_charges': np.float})
    
    #to prevent pesky spaces in column names
    telco = telco.replace(" ", "_", regex=True)
    
    telco.internet_service_type_id.loc[telco.internet_service_type_id==3] = 0 #I'll drop this later
    telco.payment_type_id.loc[telco.payment_type_id==4] = 0
    telco.contract_type_id.loc[telco.contract_type_id==3] = 0
    telco.payment_type_id = telco.payment_type_id.astype(object)
    telco.contract_type_id = telco.contract_type_id.astype(object)
    telco.internet_service_type_id = telco.internet_service_type_id.astype(object)
    
    #convert all 15 object types into dummies
    telco = pd.get_dummies(telco, drop_first=True)
    telco = telco.rename(columns={"churn_Yes": "churn", "gender_Male": "male",
                                  'internet_service_type_id_1': "dsl",
                                  'internet_service_type_id_2': "fiber_optic",
                                  'contract_type_id_1':"one_month",
                                  'contract_type_id_2':"one_year",
                                  'payment_type_id_1': "e_check",
                                  'payment_type_id_2': "check",
                                  'payment_type_id_3': "bank_transfer",
                                  'partner_Yes':"partner", 'dependents_Yes':"dependents",
                                  'phone_service_Yes':"phone_service",
                                  'multiple_lines_Yes':"multiple_lines",
                                  'online_security_Yes':"online_security",
                                  'online_backup_Yes':"online_backup",
                                  'device_protection_Yes':"device_protection",
                                  'tech_support_Yes':"tech_support",
                                  'streaming_tv_Yes':"streaming_tv",
                                  'streaming_movies_Yes':"streaming_movies",
                                  'paperless_billing_Yes':"paperless"})
    
    return pd.concat([cid, telco], axis=1)


def telco_eda_prep():
    
    telco = get_telco_data()
    
    telco = telco.drop(['customer_id', 'senior_citizen'], axis=1)
    
    #before making total charges a float, eliminate this annoying space
    telco = telco.replace(" ", 0)
    
    telco = telco.replace("No internet service", "No")
    telco = telco.replace("No phone service", "No")
    
    telco = telco.astype({'total_charges': np.float})
    
    #to prevent pesky spaces in column names
    telco = telco.replace(" ", "_", regex=True)
    
    telco.internet_service_type_id.loc[telco.internet_service_type_id==1] = "DSL"
    telco.payment_type_id.loc[telco.payment_type_id==1] = "e-check"
    telco.contract_type_id.loc[telco.contract_type_id==1] = "month-to-month"
    
    telco.internet_service_type_id.loc[telco.internet_service_type_id==2] = "fiber optic"
    telco.payment_type_id.loc[telco.payment_type_id==2] = "check"
    telco.contract_type_id.loc[telco.contract_type_id==2] = "1-year"
    
    telco.internet_service_type_id.loc[telco.internet_service_type_id==3] = "none"
    telco.payment_type_id.loc[telco.payment_type_id==3] = "bank transfer"
    telco.contract_type_id.loc[telco.contract_type_id==3] = "2-year"
    
    telco.payment_type_id.loc[telco.payment_type_id==4] = "credit card"
    
    telco['monthly_charges_bins'] = pd.qcut(telco.monthly_charges, 10)
    
    telco = telco.rename(columns={"payment_type_id": "payment_method",
                                  "contract_type_id": "contract",
                                  'internet_service_type_id': "internet_service"})
    
    return telco


if __name__ == '__main__':
    
    df = telco_model_prep()
    print(df.head())
    print(df.columns)
    df.to_csv("telco_num.csv")
    
    