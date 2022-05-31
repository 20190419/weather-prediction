

# Radwa Ahmed Elias     20190206        G3
# Mohamed Eslam Amin    20190419        G1
# omar Gamal Mohamed    20190703        G1

#=====================================================================
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

#Load a DataFrame from a CSV file
df = pd.read_csv('weatherHistory.csv')
# exclusive Selecting columns with .iloc
#dropna() Drop the rows where at least one element is missing
df = df.iloc[:, 3:11].dropna()
#returns True for every number which is less than 0.7, 70% of values are True
split = np.random.rand(len(df)) < 0.7
#return every index for which the split value is True
trainingSet = df[split]
# ~split means "not equal to" in df indexing.
testingSet = df[~split]
#number of epochs is the number of times that the entire training dataset is shown to the network during training
numberOfEpochs = 50
epochSize = int((len(trainingSet)) / numberOfEpochs)
#Randomly set the weights’ values.
weights = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
bias = random.uniform(0, 1)
learningRate = 0.0000005

epochStart = 0
epochEnd = epochSize

epoch = trainingSet.iloc[epochStart:epochEnd]
desiredOutput = epoch.iloc[:, 2]
epoch = epoch.drop('Humidity', axis='columns',)
#Return a new array S of shape:epochSize and type:float
S = np.empty(epochSize, float)
#Return a new array error of shape:epochSize and type:float.
error = np.empty(epochSize, float)
deltaWeights = np.empty((epochSize, 7), float)
deltaBias = np.empty(epochSize, float)

for k in range(numberOfEpochs):
    for i in range(epochSize):
        S[i] = np.dot(epoch.iloc[i], weights) + bias
        error[i] = S[i] - desiredOutput.iloc[i]

        deltaWeights[i] = np.dot(epoch.iloc[i], error[i] * learningRate)
        deltaBias[i] = learningRate * error[i]
    #update the weights
    weights = weights - np.mean(deltaWeights)
    #update the biase
    bias = bias - np.mean(deltaBias)

    epochStart = epochStart + epochSize
    epochEnd = epochEnd + epochSize
    epoch = trainingSet.iloc[epochStart:epochEnd]
    desiredOutput = epoch.iloc[:, 2]
    epoch = epoch.drop('Humidity', axis='columns')


epochSize = int((len(testingSet)) / numberOfEpochs)
epochStart = 0
epochEnd = epochSize

testEpoch = testingSet.iloc[epochStart:epochEnd]
realOutput = testEpoch.iloc[:, 2]
testEpoch = testEpoch.drop('Humidity', axis='columns')

meanSquareError = np.empty(numberOfEpochs, float)
predictedOutput = np.empty(epochSize, float)
E = np.empty(epochSize, float)
squareError = np.empty(epochSize, float)

#compute the MSE error
for k in range(numberOfEpochs):
    for i in range(epochSize):
        predictedOutput[i] = np.dot(testEpoch.iloc[i], weights) + bias
        E[i] = predictedOutput[i] - realOutput.iloc[i]
        squareError[i] = (E[i]) ** 2

    meanSquareError[k] = np.mean(squareError)

    epochStart = epochStart + epochSize
    epochEnd = epochEnd + epochSize
    testEpoch = testingSet.iloc[epochStart:epochEnd]
    realOutput = testEpoch.iloc[:, 2]
    testEpoch = testEpoch.drop('Humidity', axis='columns')

print(meanSquareError)
# draw a chart with a horizontal axis titled “Epoch number” ranging from 1 to 50, and a vertical axis titled “MSE”.
plt.plot(meanSquareError)
plt.suptitle('Error Calculation')
plt.show()



