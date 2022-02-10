import re
import math
import numpy as np
from hmmlearn import hmm
from itertools import product
import scipy.optimize as optimize
from sklearn.metrics import classification_report
from pyswarms.single.global_best import GlobalBestPSO

realTokens = dict()
fakeTokens = dict()
realTokensUniquePerPost = dict()
fakeTokensUniquePerPost = dict()
realTokenCount = 0
fakeTokenCount = 0
realTokenCountU = 0
fakeTokenCountU = 0
fakePosts = 0
realPosts = 0

#Read in the training file intially
csv = open('/Users/ecnu/Desktop/ThesisResearch/implementations/trainSegmented.csv', newline='')
count = 0
for l in csv:
    count += 1
    line = l.split('\t')
    text = line[1].split(' ')
    uniqueText = set(text)
    if line[0][-4:] == 'real':
        realPosts += 1 #increment the count of true posts
        for t in text:  #count all tokens in a post
            realTokenCount += 1
            if t not in realTokens.keys():
                realTokens[t] = 1
            else:
                realTokens[t] += 1
        for t in uniqueText: #count only unique tokens in a post
            realTokenCountU += 1
            if t not in realTokensUniquePerPost.keys():
                realTokensUniquePerPost[t] = 1
            else:
                realTokensUniquePerPost[t] += 1
    else:
        fakePosts += 1 #increment the count of fake posts
        for t in text: #count all tokens in a post
            fakeTokenCount += 1
            if t not in fakeTokens.keys():
                fakeTokens[t] = 1
            else:
                fakeTokens[t] += 1
        for t in uniqueText: #count only unique tokens in a post
            fakeTokenCountU += 1
            if t not in fakeTokensUniquePerPost.keys():
                fakeTokensUniquePerPost[t] = 1
            else:
                fakeTokensUniquePerPost[t] += 1
print('processed ' + str(count) + ' lines')
csv.close()

def MI(term, category):
    #instances of term in each category
    if term in realTokens.keys():
        termCountReal = realTokens[term] #instances of term in category real
    else:
        termCountReal = 0

    if term in fakeTokens.keys():
        termCountFake = fakeTokens[term] #instances of term in category fake
    else:
        termCountFake = 0

    totalTermCount  = termCountReal + termCountFake #total instances of term (specific token) in both categories YES
    totalTokenCount = realTokenCount + fakeTokenCount #total of all tokens YES
    totalPostCount = realPosts + fakePosts #total of all posts YES
    probabilityTerm = totalTermCount / totalTokenCount #probability of seeing term regardless of category YES
    probabilityTermGivenCategory = 0
    if totalTermCount == 0:
        return 0

    if category == 'real':
        probabilityCategory = realPosts/totalPostCount
        probabilityTermGivenCategory = (termCountReal / totalTokenCount) / probabilityCategory
    elif category == 'fake':
        probabilityCategory = fakePosts/totalPostCount
        probabilityTermGivenCategory = (termCountFake / totalTokenCount) / probabilityCategory
    else:
        print('wrong category')
        return 0

    if probabilityTermGivenCategory == 0:
        return 0

    return math.log2(probabilityTermGivenCategory / probabilityTerm)

def CHI(term, category):
    A = 0
    B = 0
    C = 0
    D = 0
    N = 0
    if category == 'real':
        if term in realTokensUniquePerPost.keys():
            A = realTokensUniquePerPost[term]
        if term in fakeTokensUniquePerPost.keys():
            B = fakeTokensUniquePerPost[term]
        C = realPosts - A
        D = fakePosts - B
    elif category == 'fake':
        if term in fakeTokensUniquePerPost.keys():
            A = fakeTokensUniquePerPost[term]
        if term in realTokensUniquePerPost.keys():
            B = realTokensUniquePerPost[term]
        C = fakePosts - A
        D = realPosts - B
    else:
        print('wrong category')
        return 0

    N = fakePosts + realPosts
    numerator = pow(A * D - B * C, 2) * N
    denominator = (A + B) * (C + D) * (A + C) * (B + D)
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def TFIDF(term,category):
    N = fakePosts + realPosts
    tokensReal = 0
    tokensFake = 0

    if term in realTokensUniquePerPost.keys():
        tokensReal = realTokensUniquePerPost[term]
    if term in fakeTokensUniquePerPost.keys():
        tokensFake = fakeTokensUniquePerPost[term]

    totalTokenCount = tokensReal + tokensFake

    if totalTokenCount != 0:
        if category == 'real':
            TF = tokensReal / totalTokenCount
        elif category == 'fake':
            TF = tokensFake / totalTokenCount
        else:
            print('wrong category')
            return 0

        IFD = math.log2(N/totalTokenCount + 0.01)
    else:
        #print('char not found')
        return 0

    return TF * IFD
def ECE(term, category):
    #instances of term in each category
    if term in realTokens.keys():
        termCountReal = realTokens[term] #instances of term in category real
    else:
        termCountReal = 0

    if term in fakeTokens.keys():
        termCountFake = fakeTokens[term] #instances of term in category fake
    else:
        termCountFake = 0

    totalTermCount  = termCountReal + termCountFake #total instances of term (specific token) in both categories YES
    totalTokenCount = realTokenCount + fakeTokenCount #total of all tokens YES
    totalPostCount = realPosts + fakePosts #total of all posts YES
    probabilityTerm = totalTermCount / totalTokenCount #probability of seeing term regardless of category YES
    probabilityCategoryGivenTerm = 0
    if totalTermCount == 0:
        return 0

    if category == 'real':
        probabilityCategoryGivenTerm = (termCountReal / totalTokenCount) / probabilityTerm
        probabilityCategory = realPosts/totalPostCount
    elif category == 'fake':
        probabilitynCategoryGivenTerm = (termCountFake / totalTokenCount) / probabilityTerm
        probabilityCategory = fakePosts/totalPostCount
    else:
        print('wrong category')
        return 0

    if probabilityCategoryGivenTerm == 0:
        return 0


    return probabilityCategoryGivenTerm * math.log2(probabilityCategoryGivenTerm / probabilityCategory)

def IG(term):
    termCountReal = 0
    termCountFake = 0
    totalTokenCount = realTokenCount + fakeTokenCount #total of all tokens YES

    if term in realTokens.keys():
        termCountReal = realTokens[term] #instances of term in category real
    notTermCountReal = totalTokenCount-termCountReal

    if term in fakeTokens.keys():
        termCountFake = fakeTokens[term] #instances of term in category fake
    notTermCountFake = totalTokenCount-termCountFake

    totalTermCount  = termCountReal + termCountFake #total instances of term (specific token) in both categories YES
    totalTokenCount = realTokenCount + fakeTokenCount #total of all tokens YES
    totalPostCount = realPosts + fakePosts #total of all posts YES

    probabilityTerm = totalTermCount / totalTokenCount #probability of seeing term regardless of category YES

    probabilityNotTerm = (totalTokenCount-totalTermCount) / totalTokenCount

    probabilityCategoryReal = realPosts/totalPostCount
    probabilityCategoryFake = fakePosts/totalPostCount
    p1 =  -1 * (probabilityCategoryFake * math.log2(probabilityCategoryFake) + probabilityCategoryReal * math.log2(probabilityCategoryReal))

    p2 = 0
    probabilityCategoryGivenTermReal = 0
    probabilityCategoryGivenTermFake = 0

    if probabilityTerm > 0:
        probabilityCategoryGivenTermReal = (termCountReal / totalTokenCount) / probabilityTerm
        probabilityCategoryGivenTermFake = (termCountFake / totalTokenCount) / probabilityTerm

    if probabilityCategoryGivenTermReal != 0 and probabilityCategoryGivenTermFake != 0:
        p2 = probabilityTerm * (probabilityCategoryGivenTermReal * math.log2(probabilityCategoryGivenTermReal) + probabilityCategoryGivenTermFake * math.log2(probabilityCategoryGivenTermFake))

    probabilityCategoryGivenNotTermReal = (notTermCountReal / totalTokenCount) / probabilityNotTerm
    probabilityCategoryGivenNotTermFake = (notTermCountFake / totalTokenCount) / probabilityNotTerm
    p3 = probabilityNotTerm * (probabilityCategoryGivenNotTermReal * math.log2(probabilityCategoryGivenNotTermReal) + probabilityCategoryGivenNotTermFake * math.log2(probabilityCategoryGivenNotTermFake))

    return p1 + p2 + p3

def createFeatureVector(post, category):
    mi = 0
    chi = 0
    tfidf = 0
    ece = 0
    ig = 0
    words = 0
    for t in post.split(' '):
       words += 1
       mi += MI(t, category)
       chi += CHI(t, category)
       tfidf += TFIDF(t, category)
       ece += ECE(t, category)
       ig += IG(t)
    return [mi/words, chi/words, tfidf/words, ece/words, ig/words]

#Train the HMM
def trainHMM(x, featureSelection):
    v = x.tolist()
    s = sum(featureSelection)
    print('training with PSO params:')
    print(v)

    #initialize list of vectors for real and fake
    real = list()
    fake = list()

    #Load the training file and compute feature vectors
    csv = open('/Users/ecnu/Desktop/ThesisResearch/implementations/trainSegmented.csv', newline='')
    count = 0
    for l in csv:
        line = l.split('\t')
        featureVector = createFeatureVector(line[1], line[0][-4:]) #pull all features initially
        newFeatureVector = [v1*v2 for v1,v2 in zip(featureSelection, featureVector) if v1 != 0] #zero out items not needed
        #print(newFeatureVector)
        if line[0][-4:] == 'real':
            realVector = [v1*v2 for v1,v2 in zip(newFeatureVector, v[0])]
            real.append(realVector)
        else:
            fakeVector = [v1*v2 for v1,v2 in zip(newFeatureVector, v[0])]
            fake.append(fakeVector)
    csv.close()

    startingProb = np.zeros(s)
    startingProb[0] = 1.0
    transmissionProb = np.identity(s)

    # set up fake news HMM
    modelFake = hmm.GaussianHMM(n_components=s, covariance_type="full", verbose=False)
    modelFake.startprob_ = startingProb
    modelFake.transmat_ = transmissionProb
    f = np.array(fake)
    modelFake.fit(f)

    # set up fake news HMM
    modelReal = hmm.GaussianHMM(n_components=s, covariance_type="full", verbose=False)
    modelReal.startprob_ = startingProb
    modelReal.transmat_ = transmissionProb
    r = np.array(real)
    modelReal.fit(r)
    print('vecs')
    print(r)

    accuracy = testHMM(v, featureSelection, modelReal, modelFake)
    print('initial accuracy: ' + str(accuracy))
    modelReal, modelFake = tuneHMM(v, featureSelection, modelReal, modelFake, real, fake)
    tunedAccuracy = testHMM(v, featureSelection, modelReal, modelFake)
    print('tuned accuracy: ' + str(tunedAccuracy))

    return -1 * max(accuracy, tunedAccuracy)

def testHMM(v, featureSelection, modelReal, modelFake):
    global bestAccuracy
    target_names = ['fake', 'real']
    goldStandard = list()
    predictions = list()
    print('testing beginning . . . ')
    csvTest = open('/Users/ecnu/Desktop/ThesisResearch/implementations/testSegmented.csv', newline='')
    count = 0
    correctFake = 0
    incorrectFake = 0
    correctReal = 0
    incorrectReal = 0
    for l in csvTest:
        count += 1
        line = l.split('\t')
        featureVectorR = createFeatureVector(line[1], 'real')
        newFeatureVectorR = [v1*v2 for v1,v2 in zip(featureSelection, featureVectorR) if v1 != 0]
        featureVectorF = createFeatureVector(line[1], 'fake')
        newFeatureVectorF = [v1*v2 for v1,v2 in zip(featureSelection, featureVectorF) if v1 != 0]
        fvR = np.array([v1 * v2 for v1,v2 in zip(newFeatureVectorR,v[0])])
        fvR = fvR.reshape(1, -1)
        fvF = np.array([v1 * v2 for v1,v2 in zip(newFeatureVectorF,v[0])])
        fvF = fvF.reshape(1, -1)
        try:
            fakeScore = modelFake.score(fvF)
            realScore = modelReal.score(fvR)
        except:
            print('could not score the models')
            break

        if fakeScore > realScore:
            predictions.append(0)
        else:
            predictions.append(1)

        if line[0][-4:] == 'real':
            goldStandard.append(1)
            if realScore > fakeScore:
                correctReal += 1
            else:
                incorrectReal += 1
        else:
            goldStandard.append(0)
            if fakeScore > realScore:
               correctFake += 1
            else:
               incorrectFake += 1
    accuracy = (correctReal + correctFake) / count
    if bestAccuracy < accuracy:
        print('new best accuracy:')
        bestAccuracy = accuracy
        print(featureSelection)
        print(classification_report(goldStandard, predictions, target_names=target_names))
        errorVector = list()
        for i in range(0, len(goldStandard)):
            if goldStandard[i] != predictions[i]:
                errorVector.append(i)
        print(errorVector)
        print('ensemble output')
        print(predictions)
    #print(accuracy)
    csvTest.close()
    return accuracy

def tuneHMM(v, featureSelection, modelReal, modelFake, real, fake):
    optimalEntropy = -10
    csvDev = open('/Users/ecnu/Desktop/ThesisResearch/implementations/devSegmented.csv', newline='')
    tuningFake = fake.copy()
    tuningReal = real.copy()
    print('tuning beginning . . .')
    for l in csvDev:
        line = l.split('\t')
        featureVectorR = createFeatureVector(line[1], 'real')
        newFeatureVectorR = [v1*v2 for v1,v2 in zip(featureSelection, featureVectorR) if v1 != 0]
        featureVectorF = createFeatureVector(line[1], 'fake')
        newFeatureVectorF = [v1*v2 for v1,v2 in zip(featureSelection, featureVectorF) if v1 != 0]
        fvR = np.array([v1 * v2 for v1,v2 in zip(newFeatureVectorR,v[0])])
        fvR = fvR.reshape(1, -1)
        fvF = np.array([v1 * v2 for v1,v2 in zip(newFeatureVectorF,v[0])])
        fvF = fvF.reshape(1, -1)
        try:
            fakeScore = modelFake.score(fvF)
            realScore = modelReal.score(fvR)
        except:
            print('could not score the models')
            break
        entropy = (fakeScore * math.log(abs(fakeScore)) + realScore * math.log(abs(realScore))) * -1
        #print('entropy: ' + str(entropy))
        if entropy < optimalEntropy:
            if line[0][-4:] == 'real':
                if realScore > fakeScore:
                    tuningReal.append(newFeatureVectorR)
            else:
                if fakeScore > realScore:
                    tuningFake.append(newFeatureVectorF)
    X = np.array(tuningFake)
    try:
        modelFake.fit(X)
    except:
        print('could not fit new fake model')

    #print('recreated fake news model')

    X = np.array(tuningReal)
    try:
        modelReal.fit(X)
    except:
        print('could not fit new real model')
    #print('recreated real news model')

    csvDev.close()

    return modelReal, modelFake
#initialWeights = [1, 1, 1, 1, 1]
#result = optimize.minimize(testHMM, initialWeights)

#loop through all posible combinations of features
for v in product([0,1], repeat=5):
    fVec = list(v)
    print(fVec)
    s = sum(fVec)
    if s > 0:
        bestAccuracy = 0
        x_max = np.ones(s)
        x_min = np.zeros(s)
        bounds = (x_min, x_max)
        options = {'c1': 1, 'c2': 1, 'w':1}
        optimizer = GlobalBestPSO(n_particles=1, dimensions=s, options=options, bounds=bounds)
        cost, pos = optimizer.optimize(trainHMM, 10, featureSelection=fVec)
        #print(x_max)
        #print(fVec)
        #trainHMM(x_max, fVec)
        print(cost)
        print(pos)
