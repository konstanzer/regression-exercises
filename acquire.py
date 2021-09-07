import pandas as pd
from env import host, username, password


def get_db_url(username, host, password, db):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

def get_zillow_data():
	url = get_db_url(username, host, password, 'zillow')
	query = """
		SELECT *
		FROM properties_2017
		JOIN predictions_2017 USING(parcelid)
		WHERE propertylandusetypeid
			IN (260, 261, 262, 263, 264, 265, 266, 273, 275, 276, 279)
            AND transactiondate BETWEEN '2017-05-01' AND '2017-08-31';
		"""
	return pd.read_sql(query, url)

#Link to download dataset: https://www.kaggle.com/blastchar/telco-customer-churn
def get_telco_data_kaggle():
    path = "~/Documents/Github/school/Telco/data/" #path to download location
    return pd.read_csv(path + "Kaggle_Telco.csv")

def get_db_url(username, host, password, db):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

def get_telco_data():
    url = get_db_url(username, host, password, 'telco_churn')
    query = """SELECT * FROM customers;"""
    return pd.read_sql(query, url)


if __name__ == '__main__':
    
	zillow = get_zillow_data()
	print(zillow.head())
	print(zillow.info())