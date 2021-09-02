import pandas as pd
from env import host, username, password


def get_db_url(username, host, password, db):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

def get_zillow_data():
	url = get_db_url(username, host, password, 'zillow')
	query = """
		SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
		FROM properties_2017
		WHERE propertylandusetypeid = 261
		LIMIT 10000;
		"""
	return pd.read_sql(query, url)

    
if __name__ == '__main__':
	zillow = get_zillow_data()
	print(zillow.head())
	print(zillow.info()) #891x13
	print(zillow.describe())