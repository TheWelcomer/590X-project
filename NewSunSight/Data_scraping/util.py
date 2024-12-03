# Required Libraries
from census import Census
from us import states

import pandas as pd
import csv

from geopy.geocoders import Nominatim # For address to lat/long
from uszipcode import SearchEngine
from tqdm import tqdm

from os.path import exists

from urllib.request import urlopen
import urllib
import ast


# BY ZIP LIST SOLAR DATA
def get_solar_data_by_zips(zip_codes, save=None):

    zip_codes = list(map(int, zip_codes))
    df = pd.read_csv('../Data/solar_by_zip.csv',dtype=str)

    print(len(df['region_name']))
    print(df['region_name'][0])
    print(df['region_name'][0])


    df = df[df['region_name'].isin(zip_codes)]
    df = df[['yearly_sunlight_kwh_kw_threshold_avg','number_of_panels_total','region_name']]

    if save is not None:
        df.to_csv(save+".csv", index=False)

    return df

def get_census_info_by_zip_codes(zip_codes, code_dict):

    with open('../API_keys/Census_API_key.txt', 'r') as file:
        census_api_key = str(file.read().rstrip())

    # Queries the ACS5 dataset by URL
    code_keys = str(code_dict.keys())
    url = "https://api.census.gov/data/2022/acs/acs5?get="+code_keys +"&for="
    ZCTA = 'zip code tabulation area'
    url = url + urllib.parse.quote(ZCTA) + ":*"
    url = url.replace("dict_keys(", "").replace(")", "").replace("[", "").replace("]", "").replace("'","").replace(" ","")

    # Gets Bytes from the URL, converts to str, removes null elements and then converts to list
    f = urlopen(url)
    myfile = f.read().decode("utf-8").replace("null","-1")
    res = ast.literal_eval(myfile)

    # Converts that list into a DF and then renames the columns
    df = pd.DataFrame(res[1:],columns=res[0])
    df = df.rename(columns=code_dict)
    df = df.rename(columns={'zip code tabulation area':'zcta'})
    
    # Culls to only listed zip codes
    # intersect = set(zip_codes).intersection(df['place'])
    # print(intersect)
    # print(len(intersect))

    df['zcta'].to_csv("../Data/zips.csv",index=False)

    # df = df[df['place'].isin(zip_codes)]
    return df

# Wrapper for data load calls to check params and save output, currently setup for just census, but intended to load both census and solar
def get_zip_info(zip_codes, save=None, code_dict=None):

    if code_dict is None:
        print("bad call to census dataset, no code dictionary")
        return -1
    else:
        census_df = get_census_info_by_zip_codes(zip_codes,code_dict)

        if save is not None:
            census_df.to_csv(save+".csv", index=False)

    return census_df
    # return combined

# BY BUILDING (lat/long) SOLAR DATA
def get_building_stats(lat=0,long=0,label='test',API_key=''):
    # Example lat and long
    lat = 37.4450
    long = -122.1390

    link = 'https://solar.googleapis.com/v1/buildingInsights:findClosest?location.latitude=' +str(lat)+'&location.longitude='+str(long)+'&requiredQuality=HIGH&key='+str(API_key)
    df = pd.read_json(link).drop('roofSegmentStats').drop('solarPanelConfigs').drop('financialAnalyses').drop('solarPanels').to_csv('Data/test/'+label+".csv")

    print(df)


# Define which codes to use and what they are here, find the full instructions for these codes here:
# https://www.census.gov/programs-surveys/acs/technical-documentation/code-lists.html
code_dict = {'B01003_001E': 'Total_Population',
                    'B11001_001E': 'total_households',
                    'B19013_001E': 'Median_income',
                    'B19301_001E': 'per_capita_income',
                    'B17001_002E': 'households_below_poverty_line', 
                    'B02001_003E': 'black_population',
                    'B02001_002E': 'white_population',  
                    'B02001_005E' : 'asian_population', 
                    'B02001_004E': 'native_population',}

census_df = get_zip_info(zip_codes=None, save="../Data/census_by_zip",code_dict=code_dict)
zips = pd.read_csv('../Data/zips.csv',dtype=str)
zip_codes = zips['zcta'].values
solar_df = get_solar_data_by_zips(zip_codes,save="test")


zips = zips[zips['zcta'].isin(solar_df['region_name'])]
# zips.to_csv('../Data/zips.csv', index=False)

print(len(zips))





