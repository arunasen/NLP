import random
from scipy import spatial
from sklearn.metrics import classification_report
from sklearn_extra.cluster import KMedoids
import numpy as np

with open('taggedSequencesHN.csv') as f:
    howNetSequences = f.read().splitlines()

with open('taggedSequencesANTUSD.csv') as f:
    antusdSequences = f.read().splitlines()

with open('taggedSequencesDUTIRS.csv') as f:
    dutirSequences = f.read().splitlines()
with open('taggedSequencesDUTIRSI.csv') as f:
    dutirSISequences = f.read().splitlines()
f.close()

sequences = [howNetSequences, antusdSequences, dutirSequences]
sequenceNames = ['HowNet', 'ANTUSD', 'DUTIR']
target_names = ['fake', 'real']

#faster function, non-recursive
def LCSLength(X, Y):
    m = len(X)
    n = len(Y)

    # lookup table stores solution to already computed subproblems;
    # i.e., `T[i][j]` stores the length of LCS of substring
    # `X[0…i-1]` and `Y[0…j-1]`
    T = [[0 for x in range(n + 1)] for y in range(m + 1)]

    # fill the lookup table in a bottom-up manner
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # if the current character of `X` and `Y` matches
            if X[i - 1] == Y[j - 1]:
                T[i][j] = T[i - 1][j - 1] + 1
            # otherwise, if the current character of `X` and `Y` don't match
            else:
                T[i][j] = max(T[i - 1][j], T[i][j - 1])

    # LCS will be the last entry in the lookup table
    return T[m][n];

#find a new mediod
def shortestDistanceMediod(mediodGroup, vectors, distances):
    shortestDistance = 1000000000
    newMediod = -1
    for m in mediodGroup:
        distance = 0
        for n in mediodGroup:
            distance += distances[m][n]
            if distance > shortestDistance:
                #print('breaking out')
                break
        if shortestDistance > distance:
            shortestDistance = distance
            newMediod = m
    return newMediod;

def runKMedoids(vectors, distances, goldStandard, seedK1, seedK2):
    #initially choose two mediods
    k1 = seedK1
    k2 = seedK2
    #initialize the medoids
    #while k1 == k2 or distances[k1][k2] == 0:
    #    k1 = random.randrange(0, len(vectors))
    #    k2 = random.randrange(0, len(vectors))

    newK1 = -1
    newK2 = -1
    newK1Distance = 1
    newK2Distance = 1
    count = 0
    while count != 25 and (newK1Distance != 0 or newK2Distance != 0):
        count += 1

        if newK1 != -1:
            k1 = newK1
        if newK2 != -1:
            k2 = newK2

        k1MediodGroup = {k1}
        k2MediodGroup = {k2}
        mediods = {k1, k2}
        distanceK1 = 0
        distanceK2 = 0
        for d in range(0, len(vectors)):
            if d not in mediods: #do not check against mediod itself
                distanceK1 = distances[k1][d]
                distanceK2 = distances[k2][d]

                if distanceK1 <= distanceK2:
                    k1MediodGroup.add(d)
                else:
                    k2MediodGroup.add(d)

        #calculate distances
        newK1 = shortestDistanceMediod(k1MediodGroup, vectors, distances)
        newK2 = shortestDistanceMediod(k2MediodGroup, vectors, distances)
        newK1Distance = distances[k1][newK1]
        newK2Distance = distances[k2][newK2]
#check the accuracy
    realK1 = 0
    fakeK1 = 0
    realK2 = 0
    fakeK2 = 0
    predictions = [0] * len(goldStandard)
    for m in k1MediodGroup:
        predictions[m] = 1
        if goldStandard[m] == 1:
            realK1 += 1
        else:
            fakeK1 += 1
    print('k1 group as real')
    target_names = ['fake', 'real']
    print(classification_report(goldStandard, predictions, target_names=target_names))
    errorVector = list()
    for i in range(0, len(goldStandard)):
        if goldStandard[i] != predictions[i]:
            errorVector.append(i + offsets[i])
    print(errorVector)

    print('ensemble output')
    ensemblePredictions = list()
    for i in range(0, len(predictions)):
        if offsets[i] != 0:
            for x in range (0, offsets[i]):
                ensemblePredictions.append(-1)
        ensemblePredictions.append(predictions[i])
    print(len(ensemblePredictions))
    print(ensemblePredictions)

    predictions = [0] * len(goldStandard)
    #print('k1 real: ' + str(realK1))
    #print('k1 fake: ' + str(fakeK1))

    for m in k2MediodGroup:
        predictions[m] = 1
        if goldStandard[m] == 1:
            realK2 += 1
        else:
            fakeK2 += 1
    #print('k2 real: ' + str(realK2))
    #print('k2 fake: ' + str(fakeK2))
    print('k2 group as real')
    print(classification_report(goldStandard, predictions, target_names=target_names))
    errorVector = list()
    for i in range(0, len(goldStandard)):
        if goldStandard[i] != predictions[i]:
            errorVector.append(i + offsets[i])
    print(errorVector)

    print('ensemble output')
    ensemblePredictions = list()
    for i in range(0, len(predictions)):
        if offsets[i] != 0:
            for x in range (0, offsets[i]):
                ensemblePredictions.append(-1)
        ensemblePredictions.append(predictions[i])
    print(len(ensemblePredictions))
    print(ensemblePredictions)
    print(k1MediodGroup)
    print(k1)
    print(k2MediodGroup)
    print(k2)
#read in the test data
with open('/Users/ecnu/Desktop/ThesisResearch/implementations/testSegmented.csv') as f:
    testData = f.read().splitlines()

#create the sanity check and word vectors
uniqueWords = dict()
wordCount = 0
count = 0
for l in testData:
    count  += 1
    line = l.split('\t')
    words = line[1].split(' ')
    for w in words:
        if w not in uniqueWords.keys():
            uniqueWords[w] = wordCount
            wordCount += 1
print('loaded ' + str(count) + ' lines from test data file')

loopCounter = 0
for loopCounter in range(0, 1):
    offsets = [0] * len(testData)
    offset = 0
    goldValues = list()
    wordVectors = list()
    dummyVectors = list()
    for l in testData:
        line = l.split('\t')
        wordVector = [0] * wordCount
        dummyVector = [0] * 2
        if line[0][-4:] == 'real':
            goldValues.append(1)
            dummyVector[0] = 1
        else:
            goldValues.append(0)
            dummyVector[1] = 1
        words = line[1].split(' ')
        for w in words:
            wordVector[uniqueWords[w]] = 1
        dummyVectors.append(dummyVector)
        wordVectors.append(wordVector)

    longestDistance = 0
    k1 = -1
    k2 = -1
    #compute distances
    print('computing distances')
    distances = list()
    for x in range(0, len(dummyVectors)):
        distances.append([])
        for y in range(0, len(dummyVectors)):
            distance = np.clip(spatial.distance.cityblock(dummyVectors[x], dummyVectors[y]), 0, 1)
            distances[x].append(distance)
            if distance > longestDistance:
                k1 = x
                k2 = y
    print('class values')
    #X = np.asarray(distances)
    #print('running sklearn kMedoids')
    #kmedoids = KMedoids(n_clusters=2, metric='precomputed', max_iter=3000, init='random').fit(X) #init matters!
    #print(classification_report(goldValues, kmedoids.labels_, target_names=target_names))
    #print(classification_report(inverseGoldValues, kmedoids.labels_, target_names=target_names))
    #print('running custom kMedoids')
    runKMedoids(dummyVectors, distances, goldValues, k1, k2)

    longestDistance = 0
    k1 = -1
    k2 = -1
    print('computing distances')
    distances = list()
    for x in range(0, len(wordVectors)):
        distances.append([])
        for y in range(0, len(wordVectors)):
            distance = np.clip(spatial.distance.cityblock(wordVectors[x], wordVectors[y]), 0,1)
            distances[x].append(distance)
            if distance > longestDistance:
                k1 = x
                k2 = y
    print('word vectors')
    #X = np.asarray(distances)
    #print('running sklearn kMedoids')
    #kmedoids = KMedoids(n_clusters=2, metric='precomputed', max_iter=3000, init='random').fit(X) #init matters!
    #print(classification_report(goldValues, kmedoids.labels_, target_names=target_names))
    #print(classification_report(inverseGoldValues, kmedoids.labels_, target_names=target_names))
    #print('running custom kMedoids')
    runKMedoids(wordVectors, distances, goldValues, k1, k2)
    f.close()

    #create the DUTIR emotion vectors
    goldValues = list()
    offsets = list()
    offset = 0
    emotionVectors = list()
    emotionIntensityVectors = list()
    emotionVectorReal = [0] * 7
    emotionIntensityVectorReal = [0] * 7
    emotionVectorFake = [0] * 7
    emotionIntensityVectorFake = [0] * 7
    for l in dutirSISequences:
        line = l.split(',')
        emotionVector = [0] * 7
        emotionIntensityVector = [0] * 7
        if len(line[1]) != 0:
            offsets.append(offset)
            #offset = 0
            if line[0][-4:] == 'real':
                goldValues.append(1)
            else:
                goldValues.append(0)
            for c in line[1].split(' '):
                components = c.split(':')
                emotionIntensityVector[int(components[0])] += int(components[1])
                emotionVector[int(components[0])] += 1
                if line[0][-4:] == 'real':
                    emotionIntensityVectorReal[int(components[0])] += int(components[1])
                    emotionVectorReal[int(components[0])] += 1
                else:
                    emotionIntensityVectorFake[int(components[0])] += int(components[1])
                    emotionVectorFake[int(components[0])] += 1
            emotionVectors.append(emotionVector)
            emotionIntensityVectors.append(emotionIntensityVector)
        else:
            offset += 1
    #print(offsets)

    longestDistance = 0
    k1 = -1
    k2 = -1
    print('computing distances')
    distances = list()
    for x in range(0, len(emotionVectors)):
        distances.append([])
        for y in range(0, len(emotionVectors)):
            distance = np.clip(spatial.distance.cityblock(emotionVectors[x], emotionVectors[y]), 0, 1)
            if distance > longestDistance:
                k1 = x
                k2 = y
            distances[x].append(distance)

    print('emotion vectors')
    #X = np.asarray(distances)
    #print('running sklearn kMedoids')
    #kmedoids = KMedoids(n_clusters=2, metric='precomputed', max_iter=3000, init='random').fit(X) #init matters!
    #print(classification_report(goldValues, kmedoids.labels_, target_names=target_names))
    #print(classification_report(inverseGoldValues, kmedoids.labels_, target_names=target_names))
    #print('running custom kMedoids')
    print(emotionVectors)
    print(goldValues)
    runKMedoids(emotionVectors, distances, goldValues, k1, k2)

    longestDistance = 0
    k1 = -1
    k2 = -1
    print('computing distances')
    distances = list()
    for x in range(0, len(emotionIntensityVectors)):
        distances.append([])
        for y in range(0, len(emotionIntensityVectors)):
            distance = np.clip(spatial.distance.cityblock(emotionIntensityVectors[x], emotionIntensityVectors[y]), 0, 1)
            distances[x].append(distance)
            if distance > longestDistance:
                k1 = x
                k2 = y
    print('emotion intensity vectors')
    #X = np.asarray(distances)
    #print('running sklearn kMedoids')
    #kmedoids = KMedoids(n_clusters=2, metric='precomputed', max_iter=3000, init='random').fit(X) #init matters!
    #print(classification_report(goldValues, kmedoids.labels_, target_names=target_names))
    #print(classification_report(inverseGoldValues, kmedoids.labels_, target_names=target_names))
    #print('running custom kMedoids')
    runKMedoids(emotionIntensityVectors, distances, goldValues, k1, k2)

    realCount = sum(goldValues)
    fakeCount = len(goldValues) - realCount
    print('real emotion intensity')
    print([x/realCount for x in emotionIntensityVectorReal])
    print('real emotion')
    print([x/realCount for x in emotionVectorReal])
    print('fake emotion intensity')
    print([x/fakeCount for x in emotionIntensityVectorFake])
    print('fake emotion')
    print([x/fakeCount for x in emotionVectorFake])
    print(realCount)
    print(fakeCount)

    #create sentiment strings for HowNet, ANTUSD, and DUTIR
    #for s in range(0, len(sequences)):
    for s in range(0, 3):
        offsets = list()
        offset = 0
        longestDistance = 0
        k1 = -1
        k2 = -1
        sentimentStrings = list()
        goldValues = list()
        realPositive = 0
        realNegative = 0
        fakePositive = 0
        fakeNegative = 0
        for l in sequences[s]:
            line = l.split(',')
            if len(line[1]) != 0:
                offsets.append(offset)
                offset = 0
                if line[0][-4:] == 'real':
                    goldValues.append(1)
                    realPositive = sum([int(x) for x in line[1]])
                    realNegative = len(line[1]) - sum([int(x) for x in line[1]])
                else:
                    goldValues.append(0)
                    fakePositive = sum([int(x) for x in line[1]])
                    fakeNegative = len(line[1]) - sum([int(x) for x in line[1]])
                sentimentStrings.append(line[1])
            else:
                offset += 1

        #print(offsets)
        distances = list()
        print('computing distances')
        for x in range(0, len(sentimentStrings)):
            distances.append([])
            for y in range(0, len(sentimentStrings)):
                length = LCSLength(sentimentStrings[x], sentimentStrings[y])
                distance = 1 - (length / max(len(sentimentStrings[x]), len(sentimentStrings[y])))
                distances[x].append(distance)
                if distance > longestDistance:
                    k1 = x
                    k2 = y
        realCount = sum(goldValues)
        fakeCount = len(goldValues) - realCount
        print(sequenceNames[s])
        print('real positive')
        print(realPositive/realCount)
        print('real negative')
        print(realNegative/realCount)
        print('fake positive')
        print(fakePositive/fakeCount)
        print('fake negative')
        print(fakeNegative/fakeCount)
        #X = np.asarray(distances)
        #print('running sklearn kMedoids')
        #kmedoids = KMedoids(n_clusters=2, metric='precomputed', max_iter=3000, init='random').fit(X) #init matters!
        #print(classification_report(goldValues, kmedoids.labels_, target_names=target_names))
        #print(classification_report(inverseGoldValues, kmedoids.labels_, target_names=target_names))
        #print('running custom kMedoids')
        runKMedoids(sentimentStrings, distances, goldValues, k1, k2)

    loopCounter += 1
