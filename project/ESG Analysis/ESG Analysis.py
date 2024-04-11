import pandas as pd
import numpy as np
#loading data
rawdata = pd.read_csv("E:\Data-Analysis\project\ESG Analysis\ESGData.csv")
#overview the data 
rawdata.head()
#check the data columns
rawdata.columns

#focus on recent 30ys, so keep years >= 1984
unwant = rawdata.columns[4:28]
data = rawdata.drop(unwant, axis =1)
data.head()
data.columns
data.describe()

#in this project, I'm going to observe top 5 economic countries(5.India, 4.Japan, 3.Germany, 2.China, 1.United State).
#(https://finance.yahoo.com/news/25-largest-economies-world-2024-160551586.html)

# India
IN = data.iloc[[8178, 8179, 8180, 8186, 8191, 8194, 8195, 8209, 8223]].reset_index(drop = True)
IN = IN.rename({0:'Adjusted savings: natural resources depletion (% of GNI)', 1:'Adjusted savings: net forest depletion (% of GNI)', 2:'Agricultural land (% of land area)',
                3:'CO2 emissions (metric tons per capita)', 4:'Electricity production from coal sources (% of total)', 5:'Energy use (kg of oil equivalent per capita)',
                6:'Fertility rate, total (births per woman)', 7:'Life expectancy at birth, total (years)', 8:'Population ages 65 and above (% of total population)'})
# Japan 
JP = data.iloc[[8714, 8715, 8716, 8722, 8727, 8730, 8731, 8745, 8759]].reset_index(drop = True)
JP = JP.rename({0:'Adjusted savings: natural resources depletion (% of GNI)', 1:'Adjusted savings: net forest depletion (% of GNI)', 2:'Agricultural land (% of land area)',
                3:'CO2 emissions (metric tons per capita)', 4:'Electricity production from coal sources (% of total)', 5:'Energy use (kg of oil equivalent per capita)',
                6:'Fertility rate, total (births per woman)', 7:'Life expectancy at birth, total (years)', 8:'Population ages 65 and above (% of total population)'})
# Germany
GR = data.iloc[[7374, 7375, 7376, 7382, 7387, 7390, 7391, 7405, 7419]].reset_index(drop = True)
GR = GR.rename({0:'Adjusted savings: natural resources depletion (% of GNI)', 1:'Adjusted savings: net forest depletion (% of GNI)', 2:'Agricultural land (% of land area)',
                3:'CO2 emissions (metric tons per capita)', 4:'Electricity production from coal sources (% of total)', 5:'Energy use (kg of oil equivalent per capita)',
                6:'Fertility rate, total (births per woman)', 7:'Life expectancy at birth, total (years)', 8:'Population ages 65 and above (% of total population)'})
# China
CH = data.iloc[[5431, 5432, 5433, 5439, 5444, 5447, 5448, 5462, 5476]].reset_index(drop = True)
CH = CH.rename({0:'Adjusted savings: natural resources depletion (% of GNI)', 1:'Adjusted savings: net forest depletion (% of GNI)', 2:'Agricultural land (% of land area)',
                3:'CO2 emissions (metric tons per capita)', 4:'Electricity production from coal sources (% of total)', 5:'Energy use (kg of oil equivalent per capita)',
                6:'Fertility rate, total (births per woman)', 7:'Life expectancy at birth, total (years)', 8:'Population ages 65 and above (% of total population)'})
# United State
US = data.iloc[[15414, 15415, 15416, 15422, 15427, 15430, 15431, 15445, 15459]].reset_index(drop = True)
US = US.rename({0:'Adjusted savings: natural resources depletion (% of GNI)', 1:'Adjusted savings: net forest depletion (% of GNI)', 2:'Agricultural land (% of land area)',
                3:'CO2 emissions (metric tons per capita)', 4:'Electricity production from coal sources (% of total)', 5:'Energy use (kg of oil equivalent per capita)',
                6:'Fertility rate, total (births per woman)', 7:'Life expectancy at birth, total (years)', 8:'Population ages 65 and above (% of total population)'})


from pyspark import SparkConf, SparkContext