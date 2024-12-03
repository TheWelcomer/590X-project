from census import Census
import pd

# ONE ZIP CODE (and census object) TO CENSUS DEMO DATA
def get_census_info_with_area_code(zip_code, c, code_dict, zip_objs=None):

    # List of all codes to be queried from the census data
    code_keys = list(code_dict.keys())

    if zip_objs is not None:
        # Actual Query from census data
        demographic_data = c.acs5.state_zipcode((code_keys),  {'for': 'state'.format(zip_objs[1])}, zip_objs[0])
        df = pd.DataFrame(demographic_data)
        return df

    # Define Search engine
    search = SearchEngine()

    # Search zipcode
    zipcode = search.by_zipcode(zip_code)

    if zipcode is None:
        return None

    # extract zip code state and county details
    state_fips = zipcode.state

    # Actual Query from census data
    demographic_data = c.acs5.state_zipcode((code_keys),  {'for': 'state'.format(state_fips)}, zip_code)

    # Convert to pandas DataFrame 
    df = pd.DataFrame(demographic_data)
    return df

# LIST OF ZIP CODES TO CENSUS DEMOGRAPHIC INFO
def get_census_info_by_zip_codes(code_list, code_dict):
    
    # Create a census object using the stored API key
    with open('../API_keys/Census_API_key.txt', 'r') as file:
        census_api_key = file.read().rstrip()
    c = Census(census_api_key)

    # Repeatedly query (by zip) census data for code_dict list
    df = get_census_info_with_area_code(code_list[0], c, code_dict)
    for code in tqdm(code_list[1:]):
        new =  get_census_info_with_area_code(code, c, code_dict)
        if new is not None:
            df = pd.concat([df, new])

    # Rename columns by code_dict
    df = df.rename(columns=code_dict)

    return df


def address_to_lat_long(address, geolocator=Nominatim(user_agent="Cooper Proj")):
    try:
        location = geolocator.geocode(address)
        return (location.latitude, location.longitude)
    except:
        return None
    
