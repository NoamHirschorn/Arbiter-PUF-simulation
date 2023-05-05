import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt 
class Node:
    def __init__(self, data):
        self.data = data
        self.stay = None
        self.change = None

def initializePUF(length,width):

    randomnum = 0
    basePUF = np.zeros((length,width))
    for a in range(length):
        for b in range(width):
            randomnum = -1
            while(randomnum<=0):    
                randomnum = .25*np.random.randn(1)+1
                basePUF[a][b] = float(randomnum)
    return basePUF

def runPUF(challenge,PUF,PUFwid,withXOR,numarbs):
    ans =[0 for _ in range(numarbs)]
    for b in range(numarbs):
        ans[b] = runRow(challenge,PUF[:,[b*2,b*2+1] ])
    if(withXOR):
         newanslen = int(numarbs/2)
         newans = [0 for _ in range(newanslen)]
         for b in range(newanslen):
              newans[b] = bool(ans[b*2]) ^ bool(ans[b*2+1])
         return newans
    return ans

def runRow(challenge,rowPUF):
    sum1=0.0
    sum2=0.0
    flipped = False
    for a,dig in enumerate(challenge):
            flipped = (dig==1)
            sum1 += rowPUF[a][int(flipped)]
            sum2 += rowPUF[a][int(not flipped)]          
    if(sum1>sum2):
         return 0
    return 1
def createChallenge(challengelen):
    challenge = [0 for _ in range(challengelen)]
    for i in range(len(challenge)):
          challenge[i] = int(np.random.randint(2,size=1))
    return challenge

def createChallenges(length):
    challenges = np.zeros(((2**length),length))
    for i in range(2**length):
        
        challenges[i] = np.array(list(bin(i)[2:].zfill(length))).astype('int')
    return challenges
def makeTest(PUF,length,width,predictamount):
     challenges = createChallenges(length)
     np.random.shuffle(challenges)
     numchals,_ = challenges.shape
     training = challenges[0:int(predictamount*numchals)]
     tests = challenges[int(predictamount*numchals):]
     return training,tests
def makeDataset(training,PUF,width,withXOR,lenresponse):
    numchals,lengthofchal = training.shape
    dataset = np.zeros((numchals,lenresponse))
    for ind,challenge in enumerate(training):
        dataset[ind]= runPUF(challenge,PUF,width,withXOR,int(width/2))
    return dataset
def logisticPred(training,dataset,lenresp):
    models = [0 for _ in range(lenresp)]
    for i in range(lenresp):
        #instantiate the model
        log_regression = LogisticRegression()

        #fit the model using the training data
        log_regression.fit(training,dataset[:,i])
        models[i] = log_regression

        #MODEL DIAGNOSTICS
    return models
def testModels(tests,models,PUF,width,withXOR):
     correct=0
     total=0
     for ind,challenge in enumerate(tests):
        ans= runPUF(challenge,PUF,width,withXOR,int(width/2))
        right = True
        for ind,mod in enumerate(models):
             out = mod.predict(([challenge]))
             if(out != ans[ind] ):
                  right=False
                  break
        if(right):
             correct=correct+1
        total = total+1
     #print(correct/total)
     return correct/total
def main():
     length = 12
     width = 20
     percentseen = .4
     withXOR = True
     lenresp = int(width/2)
     arr1 = [8,12,16,20,24]
     arr2 = [.1,.2,.3,.4,.5]
     arr3 = [True,False]
     resultsT = [[0 for _ in range(len(arr2))] for _ in range(len(arr1))]
     resultsF = [[0 for _ in range(len(arr2))] for _ in range(len(arr1))]
     for ind1,i in enumerate(arr1):
        width = i
        for ind2,j in enumerate(arr2):
            percentseen = j
            for k in arr3:
                withXOR = k
                if(withXOR):
                    lenresp = int(width/4)
                else:
                     lenresp = int(width/2)
                PUF = initializePUF(length,width)
                [training,tests] = makeTest(PUF,length,width,percentseen)
                dataset = makeDataset(training,PUF,width,withXOR,lenresp)
                models = logisticPred(training,dataset,lenresp)
                ans = testModels(tests,models,PUF,width,withXOR)
                print(str(i)+" "+str(j))
                if(withXOR):
                     resultsT[ind1][ind2] = ans
                else:
                     resultsF[ind1][ind2] = ans
     plt.figure()
     plt.title("Results with XORs")
     plt.xlabel("Percentage of total combinations seen")
     plt.ylabel("Accuracy of model")
     for i in range(len(arr1)):
        plt.plot(arr2,resultsT[i],label = "ratio of width to length: " +str(round(arr1[i]/length,2)))
     plt.legend(loc="upper left")
     plt.figure()
     plt.title("Results without XORs")
     plt.xlabel("Percentage of total combinations seen")
     plt.ylabel("Accuracy of model")

     for i in range(len(arr1)):
        plt.plot(arr2,resultsF[i],label = "ratio of width to length: " +str(round(arr1[i]/length,2)))
     plt.legend(loc="upper left")
     plt.show()
main()