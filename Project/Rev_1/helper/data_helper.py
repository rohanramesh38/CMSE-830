import pandas as pd
import streamlit as st
from countryinfo import CountryInfo


@st.cache_data
def load_data_all():
    deathRateData = pd.read_csv('https://raw.githubusercontent.com/rohanramesh38/CMSE-830/main/4_HW/death_rate_and_causes.csv')

    toDrop = ['G20',
        'African Region (WHO)',
        'East Asia & Pacific (WB)',
        'Eastern Mediterranean Region (WHO)',
        'Europe & Central Asia (WB)',
        'European Region (WHO)',
        'Latin America & Caribbean (WB)',
        'Middle East & North Africa (WB)',
        'North America (WB)', 'OECD Countries',
        'Region of the Americas (WHO)', 
        'South Asia (WB)', 
        'South-East Asia Region (WHO)',
        'Sub-Saharan Africa (WB)',
        'Western Pacific Region (WHO)',
        'World',
        'World Bank High Income',
        'World Bank Low Income',
        'World Bank Lower Middle Income',
        'World Bank Upper Middle Income']

    deathRateData = deathRateData[~deathRateData['Entity'].isin(toDrop)]
        # List of countries
    countries_list =deathRateData["Entity"].unique()
    exception_countries={'Andorra': 'Europe','Bahamas': 'North America','Congo': 'Africa',"Cote d'Ivoire": 'Africa','Czechia': 'Europe','Democratic Republic of Congo': 'Africa','England': 'Europe','Eswatini': 'Africa','Gambia': 'Africa','Micronesia (country)': 'Oceania','Montenegro': 'Europe','Myanmar': 'Asia','North Macedonia': 'Europe','Northern Ireland': 'Europe','Palestine': 'Asia','Sao Tome and Principe': 'Africa','Scotland': 'Europe','United States Virgin Islands': 'North America','Wales': 'Europe'}
    # Create a dictionary to store countries and their continents
    country_continents = {}
    country_iso3={}

    nf=[]
    # Iterate through the list of countries and retrieve continent information
    for country_name in countries_list:
        
        try:
            country_info = CountryInfo(country_name)
            continent = country_info.region()
            country_continents[country_name] = continent
            country_iso3[country_name]=country_info.iso(3)
        except:
            nf.append(country_name)
            country_iso3[country_name]=None
            continue

    country_continents.update(exception_countries)


    deathRateData['Continent'] = deathRateData['Entity'].map(country_continents)
    deathRateData['ISO3'] = deathRateData['Entity'].map(country_iso3)

    return deathRateData

def data_with_filtered(deathRateData,array):
    result=deathRateData
    for column in array:
        if(len(column)==2):
            name,compare=column[0],column[1]
            if(compare=='All'):
                continue
            if(type(compare) == list):
                for _ in compare:
                    result =result[result[name]==_]
            else:
                result =result[result[name]==compare]
    return result
