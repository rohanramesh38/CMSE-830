import pandas as pd
import streamlit as st
from countryinfo import CountryInfo
import requests
import numpy as np


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
    exception_iso={'Andorra': 'AND','Bahamas': 'BHS','Congo': 'COG',"Cote d'Ivoire": 'CIV','Czechia': 'CIV','Democratic Republic of Congo': 'CIV','England': 'CIV','Eswatini': 'CIV','Gambia': 'CIV','Micronesia (country)': 'CIV','Montenegro': 'CIV','Myanmar': 'MMR','North Macedonia': 'MKD','Northern Ireland': 'NIR','Palestine': 'PSE','Sao Tome and Principe': 'STP','Scotland': 'SCT','United States Virgin Islands': 'VIR','Wales': 'WAL'}
    # Create a dictionary to store countries and their continents
    country_continents = {}
    country_iso3={}

    nf=[]
    # Iterate through the list of countries and retrieve continent information
    for country_name in countries_list:
        
        try:
            country_info = CountryInfo(country_name)
            continent = country_info.region()
            country_iso3[country_name]=country_info.iso(3)
            country_continents[country_name] = continent
        except:
            nf.append(country_name)
            country_iso3[country_name]=None
            continue

    country_continents.update(exception_countries)
    country_iso3.update(exception_iso)


    deathRateData['Continent'] = deathRateData['Entity'].map(country_continents)
    deathRateData['ISO3'] = deathRateData['Entity'].map(country_iso3)
    
    #st.write(list(deathRateData['ISO3'].unique()))

    #deathRateData["Poputaltion"]=deathRateData.apply(get_pop)


    deathRateData["Poputaltion"]=deathRateData.apply(get_pop, axis=1)
    st.write(deathRateData)
    deathRateData.to_csv('parsed_1.csv', index=False)

    return deathRateData


def get_pop(row):
    #st.write(row)
    population=None
    try:
        url = f'https://api.worldbank.org/v2/country/{row["ISO3"]}/indicator/SP.POP.TOTL?date={row["Year"]}&format=json&per_page=1'
        response = requests.get(url)
        data = response.json()
        population = data[1][0]['value']
        print(row["ISO3"],row["Year"],population)
    except:
         population=None
    return population

#print(load_data_all())
def get_data_income_based():
    deathRateData = pd.read_csv('https://raw.githubusercontent.com/rohanramesh38/CMSE-830/main/4_HW/death_rate_and_causes.csv')
    Income_Groups = [
    'World Bank Low Income',
    'World Bank Lower Middle Income',
    'World Bank Upper Middle Income',
    'World Bank High Income'
    ]
    causes_of_death = deathRateData.columns.drop(['Year','Entity','Code'])

    data=deathRateData[ deathRateData.Entity.isin( Income_Groups )].melt( id_vars=['Entity', 'Year'], value_vars=causes_of_death, var_name='cause', value_name='deaths')
    print(data)
    deaths_by_income = deathRateData[ deathRateData.Entity.isin( Income_Groups )].melt( id_vars=['Entity', 'Year'], value_vars=causes_of_death, var_name='cause', value_name='deaths').drop( columns='Year').groupby( ['Entity','cause'], as_index=False).agg(np.sum)
    #print(deaths_by_income["Entity"].unique())

    return deaths_by_income
print(get_data_income_based() )