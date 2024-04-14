from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# create SparkConf - localhost
conf = SparkConf().setMaster("local").setAppName("ESG Analysis")
# create SparkSession
spark = SparkSession.builder \
        .config(conf = conf) \
        .getOrCreate()

# loading data
rawdata = spark.read.csv("E:/Data-Analysis/project/ESG Analysis/ESGData.csv", header=True, inferSchema=True)
# overview the data 
rawdata.head()
# check the data columns
rawdata.columns

# focus on recent 30ys, so keep years >= 1984
rawdata = rawdata.drop("Country Code","Indicator Code","1960","1961","1962","1963","1964","1965","1966","1967","1968","1969","1970","1971","1972","1973","1974","1975","1976","1977","1978","1979","1980","1981","1982","1983","2050")
rawdata = rawdata.drop("_c66")
rawdata.show(3)

#In this project, I'm going to observe top 5 economic countries(5.India, 4.Japan, 3.Germany, 2.China, 1.United State).
#(https://finance.yahoo.com/news/25-largest-economies-world-2024-160551586.html)
# India
IN = rawdata.filter(rawdata["Country Name"] == "India")
# Japan
JP = rawdata.filter(rawdata["Country Name"] == "Japan")
# Germany
GR = rawdata.filter(rawdata["Country Name"] == "Germany")
# China
CH = rawdata.filter(rawdata["Country Name"] == "China")
# United State
US = rawdata.filter(rawdata["Country Name"] == "United States")
# merge 5 countries data
merged_df = IN.union(JP).union(GR).union(CH).union(US)
# 9 indicators I want to process
indicator_names = ['Adjusted savings: natural resources depletion (% of GNI)', 
                   'Adjusted savings: net forest depletion (% of GNI)',
                   'Agricultural land (% of land area)',
                   'CO2 emissions (metric tons per capita)', 
                   'Electricity production from coal sources (% of total)',
                   'Energy use (kg of oil equivalent per capita)', 
                   'Fertility rate, total (births per woman)',
                   'Life expectancy at birth, total (years)', 
                   'Population ages 65 and above (% of total population)']
df = merged_df.filter(col("Indicator Name").isin(indicator_names))
df.show(3)

import matplotlib.pyplot as plt
pandas_df = df.toPandas()
# plot
for indicator_name in indicator_names:
    # 設置新的圖形
    plt.figure(figsize=(6, 6))
    plt.title(indicator_name)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.grid(True)

    for index, row in pandas_df.iterrows():
        # 提取國家名稱和指標數據
        country = row['Country Name']
        values = [row[str(year)] for year in range(1984, 2021)]
        # print
        if row['Indicator Name'] == indicator_name:
            plt.plot(range(1984, 2021), values, label=country)
    
    # 添加圖例
    plt.legend()
    plt.show()

#### TEST ####

# 過濾出 Indicator Name 為 CO2 emissions 的記錄
co2_emissions_df = df.filter(col("Indicator Name") == "CO2 emissions (metric tons per capita)")
co2_emissions_df.show(3)
# 定義要保留的列，除了年份之外的其他列
id_cols = ['Country Name', 'Indicator Name']

# 將每個年份的CO2排放量轉換為單獨的行
co2_emissions_df_Expr = "stack(38, '1984', `1984`, '1985', `1985`, '1986', `1986`, '1987', `1987`, '1988', `1988`, '1989', `1989`, '1990', `1990`, '1991', `1991`, '1992', `1992`,\
                                   '1993', `1993`, '1994', `1994`, '1995', `1995`, '1996', `1996`, '1997', `1997`, '1998', `1998`, '1999', `1999`, '2000', `2000`, '2001', `2001`,\
                                   '2002', `2002`, '2003', `2003`, '2004', `2004`, '2005', `2005`, '2006', `2006`, '2007', `2007`, '2008', `2008`, '2009', `2009`, '2010', `2010`,\
                                   '2011', `2011`, '2012', `2012`, '2013', `2013`, '2014', `2014`, '2015', `2015`, '2016', `2016`, '2017', `2017`, '2018', `2018`, '2019', `2019`, '2020', `2020`)\
                        as (Year, CO2_emissions)"
co2_emissions_df = df.select('Country Name', F.expr(co2_emissions_df_Expr))
co2_emissions_df.show(3)

### CO2 預測 ###
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import lit

# Prepare features
assembler = VectorAssembler(inputCols= ['CO2_emissions'],
                            outputCol="features")
data = assembler.transform(co2_emissions_df)
data.printSchema()

data = data.withColumn("Year", data["Year"].cast("double"))
# 更新特徵向量內容，包含 Year
assembler = VectorAssembler(inputCols=['Year', 'CO2_emissions'], outputCol="features")
data = assembler.transform(data)


# 預測 CO2 emissions
predictions = lr_model.transform(data)
predictions.show()
# 創建一個空的 DataFrame 來存儲預測結果
prediction_df = spark.createDataFrame([], co2_emissions_df.schema)
# 遍歷 2021 年至 2030 年的每一年
for year in range(2021, 2031):
    # 準備測試數據
    test_data = data.withColumn("Year", lit(year))
    
    # 訓練模型
    lr = LinearRegression(featuresCol="features", labelCol="CO2_emissions", predictionCol=f"predicted_CO2_{year}")
    lr_model = lr.fit(data)
    
    # 預測並添加到結果 DataFrame 中
    predictions = lr_model.transform(test_data)
    prediction_df = prediction_df.union(predictions.select("Country Name", "Year", f"predicted_CO2_{year}"))


# stop SparkSession
spark.stop()