
from operator import index
from tokenize import group
from unicodedata import numeric
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import re
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

import seaborn as sns 
from keras.layers import LSTM,Dense
from keras import models
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1066634)





    
def getDirectories(path):
    """
    It takes a path as an argument and returns a list of all the files in that path
    
    :param path: the path to the directory containing the data
    :return: A list of all the files in the directory
    """
    demandDir = []

    for root,dirs,files in os.walk(path):
        for dataset in files:
            temp = os.path.join(root,dataset)
            if os.stat(temp).st_size == 0:
                continue
            demandDir.append(temp)
               
    return demandDir


def mergeFrames(dirList,cols):
    """
    It takes a list of file paths, and a list of column names, and returns a dataframe with the data
    from all the files in the list
    
    :param dirList: a list of all the files in the directory
    :param cols: the columns of the dataframe
    :return: A dataframe with the columns specified in the cols list.
    """
    temp = pd.DataFrame(columns=cols)
    for dataset in dirList:
        df = pd.read_csv(dataset)
        df.rename(columns=lambda x: x.lower(),inplace=True)
        df.rename(columns=lambda x: x.title(),inplace=True)
        df.drop_duplicates(subset=["Time"], keep="first",inplace=True)
        if len(df)!=288:
            continue   
        
        name = nameREGEX.findall(dataset)[0]
        try:
            year = int(name[:4])
            month = int(name[4:6])
            day = int(name[6:8])
            date = datetime(year,month,day)
            finalDate = date.strftime("%Y-%m-%d")

        except Exception:
            continue
        df["Datetime"] = finalDate
        temp = pd.concat([temp, df])
    
    return temp

def findDailyMomentsDemands(df):
    temp = pd.DataFrame(columns=["Mean","Variance","Skewness","Kurtosis"])
    meanDemands = df.groupby("Datetime").mean()["Current Demand"]
    varDemands = df.groupby("Datetime").var()["Current Demand"]
    skewDemands = df.groupby("Datetime").apply(pd.DataFrame.skew,numeric_only=True)["Current Demand"]
    kurtDemands = df.groupby("Datetime").apply(pd.DataFrame.kurt,numeric_only=True)["Current Demand"]

    temp["Mean"] = meanDemands
    temp["Variance"] = varDemands
    temp["Skewness"] = skewDemands
    temp["Kurtosis"] = kurtDemands

    return temp.reset_index()

def findMonthMoments(df,dates,months=3):
    """
    It takes a dataframe, a list of dates, and the number of months to be considered. It then creates a
    new dataframe with the mean, variance, skewness, and kurtosis of the dataframe for each date in the
    list
    
    :param df: The dataframe that contains the data
    :param dates: A list of dates in the format "YYYY-MM"
    :param months: The number of months to consider for each datetime, defaults to 3 (optional)
    :return: A dataframe with the mean, variance, skewness, and kurtosis of the demand for each month.
    """
    counter = 0
    final = pd.DataFrame(columns=["Mean","Variance","Skewness","Kurtosis","Datetime"])
    temp = pd.DataFrame(columns=df.columns)
    for date in dates:
        temp = pd.concat([temp, df.loc[df["Datetime"].str.contains(date)]])
        counter+=1
        if counter == months:
            meanVal = temp["Current Demand"].mean(axis=0)
            varVal = temp["Current Demand"].var(axis=0)
            skewVal = temp["Current Demand"].skew(axis=0)
            kurtVal = temp["Current Demand"].kurtosis(axis=0)
            final = pd.concat([final,pd.DataFrame({"Mean":[meanVal],"Variance":[varVal],"Skewness":[skewVal],"Kurtosis":[kurtVal],"Datetime":[date]})], ignore_index=True)
            temp = pd.DataFrame(columns=df.columns)
            counter =0
    
    return final


def findDailyMomentsSources(df):
    """
    It takes a dataframe and returns the mean, variance, skewness, and kurtosis of each column, grouped
    by the datetime column
    
    :param df: dataframe
    :return: The mean, variance, skew, and kurtosis of the dataframe grouped by datetime.
    """

    meanSources = df.groupby("Datetime").mean()
    varSources = df.groupby("Datetime").var()
    skewSources = df.groupby("Datetime").apply(pd.DataFrame.skew,numeric_only=True)
    kurtSources = df.groupby("Datetime").apply(pd.DataFrame.kurt,numeric_only=True)

    return meanSources,varSources,skewSources,kurtSources



def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def removeNonMatchingDates(demands,sources,dates):
    """
    It takes in the demands dataframe, the sources dataframe, and the dates list. It then creates a set
    of the dates in the sources dataframe. It then iterates through the dates list and checks if each
    date is in the set. If it is not, it adds it to a list of excluded dates. Finally, it iterates
    through the excluded dates list and drops the rows in the demands dataframe that have those dates
    
    :param demands: the dataframe of demands
    :param sources: the dataframe containing the source data
    :param dates: a list of datetime objects
    """
    temp = set(sources["Datetime"])
    excludedDates = []
    for date in dates:
        if date not in temp:
            excludedDates.append(date)
    for ex in excludedDates:
        demands.drop(demands.index[demands["Datetime"]==ex],inplace=True)

def findDuckCurve(demands,sources):
    """
    It takes in the demand dataframe and the sources dataframe, and returns a groupby object of the
    demand dataframe with the solar data subtracted from it
    
    :param demands: a dataframe of the demand data
    :param sources: a dictionary of dataframes, each containing the data for a source of energy
    :return: A groupby object
    """
    solar = sources["Solar"].reset_index(drop =True).copy()
    curDemand = demands[["Current Demand","Datetime"]].copy()
    
    temp = curDemand["Current Demand"] - solar
    curDemand["Demand Without Solar"] = temp

    groups = curDemand.groupby("Datetime")
    
    return groups

def plotDuck(groupedValues, date):
    """
    It takes in a dataframe, and a date, and plots the current demand and demand without solar for that
    date
    
    :param groupedValues: the dataframe grouped by date
    :param date: The date you want to plot
    """
    grp = groupedValues.get_group(date).reset_index(drop =True)
    plt.plot(grp["Current Demand"])
    plt.plot(grp["Demand Without Solar"])
    plt.title(date)


# create lists with the directories of every dataset in the demands
# and sources folders and sorts them
demandPath = "./demand"
sourcesPath = "./sources"
nameREGEX = re.compile(r'[0-9]+')
demandDir = sorted(getDirectories(demandPath))
sourcesPath = sorted(getDirectories(sourcesPath))

# lists with the column names of the merged dataframes to be created
colsDemands =["Day Ahead Forecast","Hour Ahead Forecast","Current Demand","Datetime"]
colsSources =["Time","Solar","Wind","Geothermal","Biomass","Biogas","Small Hydro","Coal","Nuclear","Natural Gas",\
                "Large Hydro","Batteries","Imports","Other","Datetime"]

# get the dates of every day in the dataset
datesDemands = [nameREGEX.findall(x)[0] for x in demandDir]
datesDemands = [f'{x[:4]}-{x[4:6]}-{x[6:8]}' for x in datesDemands]
# get the monts of every dataset
months = [x[:-3] for x in datesDemands]
months = sorted(list(set(months)))

# merge the dataframes fill all null values with 0 and reset the index 
# in order to have a common index for every 5 minutes of the day
mergedDemands = mergeFrames(demandDir,colsDemands).reset_index(drop =True).dropna()
mergedSources = mergeFrames(sourcesPath,colsSources).reset_index(drop =True).dropna()

removeNonMatchingDates(mergedDemands,mergedSources,datesDemands)
mergedDemands = mergedDemands.reset_index(drop =True)
mergedSources = mergedSources.reset_index(drop =True)


# store all the csvs for easier manipulation and time saving
mergedDemands.to_csv("./MergedDemands.csv")
mergedSources.to_csv("./MergedSources.csv")


mergedDemands = pd.read_csv("./MergedDemands.csv", index_col= 0)
mergedSources = pd.read_csv("./MergedSources.csv", index_col= 0)

# get all the mathematical moments of sources and demands(mean, variance)
# skewness and kurtosis. for the demands get the mathematical moments
# for a 3-month, 6-month and 12-month period 
dailyMomentsDemands = findDailyMomentsDemands(mergedDemands)
quarterMoments = findMonthMoments(mergedDemands,months,3)
halfyearMoments = findMonthMoments(mergedDemands,months,6)
yearlyMoments = findMonthMoments(mergedDemands,months,12)
dailyMomentsSources = findDailyMomentsSources(mergedSources)

dailyMomentsSources[0].to_csv("./MeanSources.csv")
dailyMomentsSources[1].to_csv("./VarSources.csv")
dailyMomentsSources[2].to_csv("./SkewSources.csv")
dailyMomentsSources[3].to_csv("./KurtSources.csv")
dailyMomentsDemands.to_csv("./DemandMoments.csv")



dailyMomentsDemands = pd.read_csv("./DemandMoments.csv", index_col= 0).reset_index()
meanSources = pd.read_csv("./MeanSources.csv", index_col= 0).reset_index()
varSources = pd.read_csv("./VarSources.csv", index_col= 0).reset_index()
kurtSources = pd.read_csv("./KurtSources.csv", index_col= 0).reset_index()
skewSources = pd.read_csv("./SkewSources.csv", index_col= 0).reset_index()

# # get the grouby object of the duckcurve of every day 
groups=findDuckCurve(mergedDemands,mergedSources)

#--------------------------------------------COMMENT THIS BLOCK IF NOT USING QT BACKEND----------------------------------------#
#--------------------------------------------COMMENT THIS BLOCK IF NOT USING QT BACKEND----------------------------------------#
#--------------------------------------------COMMENT THIS BLOCK IF NOT USING QT BACKEND----------------------------------------#
#--------------------------------------------COMMENT THIS BLOCK IF NOT USING QT BACKEND----------------------------------------#
#--------------------------------------------COMMENT THIS BLOCK IF NOT USING QT BACKEND----------------------------------------#
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
#------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------#

plt.figure(1)
plt.plot(mergedSources.sum(axis=1, numeric_only=True))
plt.plot(mergedDemands["Current Demand"])

days = 30
labels = sorted(list(set(mergedDemands["Datetime"])))[0::days]
dayTicks = [i*288*days for i in range(len(labels))]
plt.xticks(dayTicks, labels ,rotation=30)
plt.title("Demand and Energy Production every 5 minutes")
plt.xlabel("Dates(Every 30 days)",fontweight="bold")
plt.ylabel("Energy Units",fontweight="bold")
plt.ylim([-10_000,50_000])
plt.legend(["Production","Demands"])


plt.figure(2)
days = 45
labels = dailyMomentsDemands["Datetime"][0::days]
ticks = [x for x in range(0,len(dailyMomentsDemands["Datetime"]),days)]

plt.subplot(2,2,1)
plt.plot(dailyMomentsDemands["Mean"])
plt.xticks(ticks, labels ,rotation=17,fontsize = 6)
plt.title("Mean Daily Values of Demand")
plt.subplot(2,2,2)
plt.plot(dailyMomentsDemands["Variance"])
plt.xticks(ticks, labels ,rotation=17,fontsize = 6)
plt.title("Variance Daily Values of Demand")
plt.subplot(2,2,3)
plt.plot(dailyMomentsDemands["Skewness"])
plt.xticks(ticks, labels ,rotation=17,fontsize = 6)
plt.title("Skewness Daily Values of Demand")
plt.subplot(2,2,4)
plt.plot(dailyMomentsDemands["Kurtosis"])
plt.title("Kurtosis Daily Values of Demand")
plt.xticks(ticks, labels ,rotation=17,fontsize = 6)

plt.figure(3)
days = 45
plt.subplot(2,2,1)
plt.plot(meanSources.sum(axis=1, numeric_only=True))
plt.xticks(ticks, labels ,rotation=17,fontsize = 6)
plt.title("Mean Daily Values of Sources")
plt.subplot(2,2,2)
plt.plot(varSources.sum(axis=1, numeric_only=True))
plt.xticks(ticks, labels ,rotation=17,fontsize = 6)
plt.title("Variance Daily Values of Sources")
plt.subplot(2,2,3)
plt.plot(skewSources.sum(axis=1, numeric_only=True))
plt.xticks(ticks, labels ,rotation=17,fontsize = 6)
plt.title("Skewness Daily Values of Sources")
plt.subplot(2,2,4)
plt.plot(kurtSources.sum(axis=1, numeric_only=True))
plt.title("Kurtosis Daily Values of Sources")
plt.xticks(ticks, labels ,rotation=17,fontsize = 6)

plt.figure(7)
for i in range(8):
    plt.subplot(2,4,i+1)
    plotDuck(groups,np.random.choice(datesDemands))
plt.xlabel("Time(5min)")
plt.ylabel("Energy Units")

# create a dataframe with the mean demands and sources as well as datetime
df = pd.DataFrame(columns=["Demands","Sources","Datetime"])
df["Demands"] = dailyMomentsDemands["Mean"]
df["Sources"] = meanSources.loc[:,meanSources.columns!="Datetime"].sum(axis =1)
df["Datetime"] = meanSources["Datetime"]

clusterFrame = df.loc[:,df.columns!="Datetime"].dropna()

nrstNeighbors = NearestNeighbors().fit(clusterFrame)
neighDistance, neighInd = nrstNeighbors.kneighbors(clusterFrame)
sortedNeighborDistance = np.sort(neighDistance, axis=0)
kDistance = sortedNeighborDistance[:, 4]

# using the elbow method find the ε for the dbscan
plt.figure(4)
plt.plot(kDistance)
plt.axhline(y=450, linewidth=1, linestyle='dashed')
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")


#find the clusters with DBSCAN
# we have 2 features so samples 2*2 = 4
clusters = DBSCAN(eps=450, min_samples=4).fit(clusterFrame)

# scatterplot for the sources and demands of each day
plt.figure(5)
p = sns.scatterplot(data=clusterFrame, x="Sources", y="Demands", hue=clusters.labels_, legend="full")
sns.move_legend(p, "upper right", title='Clusters')


# find the outlouers and get the dates of the outlier dates
outliers = clusterFrame[clusters.labels_ == -1]
outlierDates = []

for i,j in zip(outliers.Demands,outliers.Sources):
    outlierDates.append(df.loc[(abs(clusterFrame['Demands']-i)<10e-5) & (abs(df['Sources']-j)<10e-5)]["Datetime"].values[0])
# remove the outliers
newMergedDemands = mergedDemands[mergedDemands["Datetime"].isin(outlierDates) == False]
newMergedSources = mergedSources[mergedSources["Datetime"].isin(outlierDates) == False]

# create the dataset conaining only the renewable sources
fossilSources = newMergedSources[["Coal","Nuclear","Natural Gas", "Batteries", "Imports", "Other"]]
fossilSources = fossilSources.sum(axis=1)
renewableSources = (newMergedDemands["Current Demand"] - fossilSources ).to_frame().dropna()[:2*365*288]

# split the data into train and test set
trainSize = int(len(renewableSources) * 0.75)
testSize = int(len(renewableSources) * 0.25)
trainSet = renewableSources.iloc[:trainSize]
testSet = renewableSources.iloc[trainSize:]

# min max scale transform the data
dataScaler = MinMaxScaler()
dataScaler.fit(trainSet)
scaledTrain = dataScaler.transform(trainSet)
scaledSet = dataScaler.transform(testSet)


# create the lstm newtork
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
inputNum = 4
featuresNum = 1
generator = TimeseriesGenerator(scaledTrain, scaledTrain, length=inputNum, batch_size=16)

model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(inputNum, featuresNum)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(generator,epochs=15, batch_size= 16,verbose=2)
model.save("./LSTM_model")

model = models.load_model("./LSTM_model")
testPredictions = model.predict(scaledSet)
truePredictions = dataScaler.inverse_transform(testPredictions)
test = np.zeros(len(scaledTrain))
test = np.concatenate((test,truePredictions),axis=None)

plt.figure(6)
plt.plot(renewableSources.sum(axis=1,numeric_only=True).reset_index(drop=True))
plt.plot(test)
plt.xlabel("")
plt.title("Prediction and Actual values for the renewable energy")
plt.xlabel("Time axis(5min)",fontweight="bold")
plt.ylabel("Energy Units",fontweight="bold")
plt.ylim([-1000,50_000])
plt.legend(["Actual Value","Predicted"])
plt.show()





