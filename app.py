import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from fbprophet import Prophet
import streamlit as st
#import warnings
#warnings.filterwarnings('ignore')
def main():

    st.title("Forecasting prices of Avocados 🥑 for next 365 days")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('avocado.csv')
        #data.head()
        return data

    avocado_df = load_data()
    avocado_df = avocado_df.sort_values("Date")
    st.write(avocado_df)

    select = st.selectbox('Select option',['Select','Visualizations','Prediction (all regions)','Prediction (region specific)'], key='3')
    if select=='Visualizations':
        select1 = st.selectbox('Select the visulization',['Select','Date vs Average Price','Distribution of Average price','Violin plot of Average price vs Avocado','Barchart on basis of regions','barchart on basis of count of year','Average price of conventional avocados in regions','Average price of organic avocados in regions'], key='3')
        #@st.cache(persist=True)
        if (select1=='Date vs Average Price'):
            def date_avg(avocado_df):
                plt.figure(figsize=(10,10))
                plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])
                plt.title("Date vs Average price")
                st.pyplot()

            date_avg(avocado_df)
        elif (select1=='Distribution of Average price'):
            #Plotting distribution of the average price
            #@st.cache(persist=True)
            def dist_avg(avocado_df):
                plt.figure(figsize=(10,6))
                sns.distplot(avocado_df["AveragePrice"], color = 'b')
                plt.title("Distribution of Average price")
                st.pyplot()

            dist_avg(avocado_df)
        elif (select1=='Violin plot of Average price vs Avocado'):
            #Violin plot of the average price vs avocado type
            #@st.cache(persist=True)
            def violion_plot(avocado_df):
                sns.violinplot(y="AveragePrice", x="type", data = avocado_df)
                plt.title("Violin plot of Average price vs Avocado")
                st.pyplot()

            violion_plot(avocado_df)

        elif (select1=='Barchart on basis of regions'):
            #barchart to indicate the number of regions
            st.write("barchart on basis of regions")
            sns.set(font_scale=0.7)
            plt.figure(figsize=[25,12])
            sns.countplot(x = 'region', data = avocado_df)
            plt.xticks(rotation = 45)
            st.pyplot()

        elif (select1=='barchart on basis of count of year'):
            # Bar Chart to indicate the count in every year
            st.write("barchart on basis of count of year")
            sns.set(font_scale=1.5)
            plt.figure(figsize=[25,12])
            sns.countplot(x = 'year', data = avocado_df)
            plt.xticks(rotation = 45)
            st.pyplot()

         #Plot of avocado prices vs. regions for conventional avocados
        elif (select1=='Average price of conventional avocados in regions'):
            st.write("Average price vs regions for conventional avocados")
            conventional = sns.catplot('AveragePrice','region', data = avocado_df[ avocado_df['type']=='conventional'],
                           hue='year',
                           height=20)
            st.pyplot()

        elif (select1=='Average price of organic avocados in regions'):
            st.write("Average price vs regions for organic avocados")
            organic = sns.catplot('AveragePrice','region', data = avocado_df[ avocado_df['type']=='organic'],
                       hue='year',
                       height=20)
            st.pyplot()

    #@st.cache(persist=True)
    if select=='Prediction (all regions)':
        st.title("Total Price prediction on basis of all regions")

        #@st.cache(persist=True)
        def preprocess(avocado_df):
            avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]
            avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds', 'AveragePrice':'y'})
            return avocado_prophet_df

        avocado_prophet_df = preprocess(avocado_df)
        with st.spinner(text="Preprocessing..."):

            m = Prophet()
            m.fit(avocado_prophet_df)
            #periods = st.number_input("No of days to predict",1,365, step=1, key='periods')
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)
        #st.write("Predicting...")
            with st.spinner(text="Predicting..."):

                figure = m.plot(forecast, xlabel='Date', ylabel='Price')
                plt.title("Forecast for next 365 for all regions collectively")
                st.pyplot()

                st.write("Trend analysis")
                m.plot_components(forecast)
                st.pyplot()

    #@st.cache(persist=True)
    if(select=='Prediction (region specific)'):
        st.title("Price prediction in specific region")
        choice = st.selectbox('Pick region', ('Albany', 'Atlanta', 'BaltimoreWashington', 'Boise', 'Boston',
           'BuffaloRochester', 'California', 'Charlotte', 'Chicago',
           'CincinnatiDayton', 'Columbus', 'DallasFtWorth', 'Denver',
           'Detroit', 'GrandRapids', 'GreatLakes', 'HarrisburgScranton',
           'HartfordSpringfield', 'Houston', 'Indianapolis', 'Jacksonville',
           'LasVegas', 'LosAngeles', 'Louisville', 'MiamiFtLauderdale',
           'Midsouth', 'Nashville', 'NewOrleansMobile', 'NewYork',
           'Northeast', 'NorthernNewEngland', 'Orlando', 'Philadelphia',
           'PhoenixTucson', 'Pittsburgh', 'Plains', 'Portland',
           'RaleighGreensboro', 'RichmondNorfolk', 'Roanoke', 'Sacramento',
           'SanDiego', 'SanFrancisco', 'Seattle', 'SouthCarolina',
           'SouthCentral', 'Southeast', 'Spokane', 'StLouis', 'Syracuse',
           'Tampa', 'TotalUS', 'West', 'WestTexNewMexico'), key=0)
        if len(choice)>0:
            with st.spinner(text="Preprocessing..."):
                avocado_df_sample = avocado_df[avocado_df['region']==choice]
                avocado_df_sample = avocado_df_sample.sort_values("Date")
                plt.figure(figsize=(10,10))
                plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])
                plt.xlabel('Price')
                plt.title("Change of average price with date in {} region".format(choice))
                st.pyplot()
                avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds', 'AveragePrice':'y'})
                m = Prophet()
                m.fit(avocado_df_sample)
                with st.spinner(text="Predicting..."):
                    # Forcasting into the future
                    future = m.make_future_dataframe(periods=365)
                    forecast = m.predict(future)
                    m.plot(forecast, xlabel='Date', ylabel='Price')
                    plt.title("Forcast of average price for next 365 days in {} region".format(choice))
                    st.pyplot()

                    st.write("Forcast on basis of trend in {} region".format(choice))
                    m.plot_components(forecast)
                    st.pyplot()




















if __name__ == '__main__':
    main()
