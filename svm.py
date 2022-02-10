import pynlpir
from spmf import Spmf
import pandas as pd
from itertools import product
from libsvm.svmutil import *
from gensim.models import Word2Vec
from sklearn.metrics import classification_report

#import the lexicon
lexicon = open('/Users/ecnu/Desktop/ThesisResearch/DUTIR/lexicon.txt', newline='\r\n')

#map subcategories to categories
emotionMapping = {'PA': 0, 'PE': 0, 'PD': 1, 'PH': 1, 'PG': 1, 'PB': 1, 'PK': 1, 'NA': 2, 'NB': 3, 'NJ': 3, 'NH': 3, 'PF': 3, 'NI': 4, 'NC': 4, 'NG': 4, 'NE': 5, 'ND': 5, 'NN': 5, 'NK': 5, 'NL': 5, 'PC': 6}
dictionary = dict()
count = 0
for l in lexicon:
    count += 1
    line = l.split('\t')
    if count != 1:
        dictionary[line[0]] = (emotionMapping[line[4].strip()], int(line[5]))
print('processed ' + str(count) + ' rows')
lexicon.close()

with open('/Users/ecnu/Desktop/ThesisResearch/implementations/trainUnSegmented.csv', newline='') as f:
    trainData = f.read().splitlines()
f.close()

uniqueWords = [0] #class placeholder
conjunctions = dict()
existingRules = set() #track rules already created
emotionVectorReal = [0] * 7
emotionVectorFake = [0] * 7
countReal = 0
countFake = 0

pynlpir.open()
f = open('rules.txt', 'w+')
c = 9 #sequencing starts after all emotion categories and class values
for l in trainData:
    rule = []
    emotionVector = [0] * 7
    line = l.split('\t')
    #print(line)
    words = pynlpir.segment(line[1])
    if line[0][-4:] == 'real':
        cls = '1'
        countReal += 1
    else:
        cls = '0'
        countFake +=  1
    wordVector = [cls]
    for w in words:
        wordVector.append(w[0])
        ruleLength = str(len(rule)) #force the sequences to be semi-unique
        if w[0] not in uniqueWords:
            uniqueWords.append(w[0])
        if w[1] == 'punctuation mark':
            maxValue = max(emotionVector)
            rule.append(str(emotionVector.index(maxValue) + 2)) #values 1 and 2 allocated already
            #rule.append(str(emotionVector.index(maxValue) + 2))
            rule.append('-1')
            emotionVector = [0] * 7
        #sequencing depends on single digit key values
        #elif w[1] == 'conjunction':
        #    if w[0] not in conjunctions.keys():
        #        conjunctions[w[0]] = c
        #        c += 1
        #    rule.append(str(conjunctions[w[0]]))
        #    rule.append('-1')
        else: #accumulate sentiment
            if w[0] in dictionary.keys():
                sentiment = dictionary[w[0]]
                emotionVector[sentiment[0]] += sentiment[1]
                if cls == '1':
                    emotionVectorReal[sentiment[0]] += sentiment[1]
                else:
                    emotionVectorFake[sentiment[0]] += sentiment[1]
    rule.append(cls)
    rule.append('-1')
    rule.append('-2')
    if len(rule) > 3:
        ruleText = ' '.join(rule)
        if ruleText not in existingRules:
            existingRules.add(ruleText)
            f.write(ruleText + '\r\n')
f.close()
#print(conjunctions)
print(len(conjunctions))
print('average real emotion vector')
print(' '.join([str(x/countReal) for x in emotionVectorReal]))
print('average fake emotion vector')
print(' '.join([str(x/countFake) for x in emotionVectorFake]))

#mine the top 15 fake news rules with confidence of 100%
spmf = Spmf('TopSeqClassRules', input_filename='rules.txt', output_filename='/Users/ecnu/Desktop/ThesisResearch/implementations/SVM/fakeRules.txt', spmf_bin_location_dir='/Users/ecnu/Desktop/ThesisResearch/implementations/SVM/test_files', arguments=[15, 1, 0])
spmf.run()
print('mined fake news rules')
#mine the top 15 real news rules with confidence of 100%
spmf = Spmf('TopSeqClassRules', input_filename='rules.txt', output_filename='/Users/ecnu/Desktop/ThesisResearch/implementations/SVM/realRules.txt', spmf_bin_location_dir='/Users/ecnu/Desktop/ThesisResearch/implementations/SVM/test_files', arguments=[50, 1, 1])
spmf.run()
print('mined real news rules')


#load the rules and remove placeholders
def loadRules(ruleFile):
    count = 0
    rules = set()
    for l in ruleFile:
        count += 1
        line = l.split(' ')
        antecendent = [str(int(x[-3:])) for x in line[0].split(',')]
        rules.add(' '.join(antecendent))
    print('processed ' + str(count) + ' rules')
    return list(rules)

realRuleFile = open('/Users/ecnu/Desktop/ThesisResearch/implementations/SVM/realRules.txt', newline='')
realRules = loadRules(realRuleFile)
#print('real rules')
#print(realRules)
#print(len(realRules))
realRuleFile.close()

fakeRuleFile = open('/Users/ecnu/Desktop/ThesisResearch/implementations/SVM/fakeRules.txt', newline='')
fakeRules = loadRules(fakeRuleFile)
#print('fake rules')
#print(fakeRules)
#print(len(fakeRules))
fakeRuleFile.close()

#Check for duplicates
for r in realRules:
    for f in fakeRules:
        if r == f:
            print('found duplicate')
            realRules.remove(r)
            fakeRules.remove(f)

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

#create a word2vec model representation of the training posts
#df = pd.read_csv('/Users/ecnu/Desktop/ThesisResearch/implementations/trainSegmented.csv')
#df1 = df.apply(lambda x: ''.join(x.astype(str)), axis=1)
#sent = [row.split('\t')[1].split(' ') for row in df1]
#model = Word2Vec(sentences=sent, vector_size=100, window=5, min_count=1, workers=4)

#create the vectors
def createVectors(file, mode, features):
    count = 0
    if mode == 'train':
        f = open('train1SVM.txt', 'w+')
    elif mode == 'test':
        f = open('test1SVM.txt', 'w+')

    for l in file:
        count += 1
        line = l.split('\t')
        words = pynlpir.segment(line[1])
        wordVector = [0] * len(uniqueWords)
        emotionVector = [0] * 7
        emotionSequence = list()
        emotionVectorSentence = [0] * 7
        rulesVector = [0] * (len(realRules) + len(fakeRules))

        if line[0][-4:] == 'real':
            cls = 1
        else:
            cls = 0
        if mode == 'test':
            goldStandard.append(cls)

        for w in words:
            if features[0] == 1:
                #Build the emotion vector
                if w[0] in dictionary.keys():
                    sentiment = dictionary[w[0]]
                    emotionVector[sentiment[0]] += sentiment[1]

            if features[1] == 1:
                #Build the rules vector
                if w[1] == 'punctuation mark':
                    maxValue = max(emotionVectorSentence)
                    emotionSequence.append(str(emotionVectorSentence.index(maxValue) + 2)) #values 1 and 2 allocated already
                    emotionVectorSentence = [0] * 7
                #elif w[1] == 'conjunction':
                #    if w[0] in conjunctions.keys():
                #        emotionSequence.append(str(conjunctions[w[0]])) #values 2 to 8 are allocated
                else:
                    if w[0] in dictionary.keys():
                        sentiment = dictionary[w[0]]
                        emotionVectorSentence[sentiment[0]] += sentiment[1]
            if features[2] == 1:
                if w[0] in uniqueWords:
                    wordVector[uniqueWords.index(w[0])] = 1
        #fire the rules
        if features[1] == 1:
            if (cls == 1 and mode == 'train') or mode == 'test':
                for r in realRules:
                    seq = ' '.join(emotionSequence)
                    cover = LCSLength(seq,r)
                    if cover == len(r):
                        rulesVector[realRules.index(r)] +=1

            if (cls == 0 and mode == 'train') or mode == 'test':
                for r in fakeRules:
                    seq = ' '.join(emotionSequence)
                    cover = LCSLength(seq,r)
                    if cover == len(r):
                        rulesVector[len(realRules) + fakeRules.index(r)] +=1
            #Build the word and punctuation vector
            #gensim
            #if w[0] in model.wv:
            #    newWordVector = list()
            #    for x,y in zip(model.wv[w[0]], wordVector):
            #        newWordVector.append(x+y)
            #    wordVector = newWordVector
                    #average out the word vector-gensim
                    #for i in range(0, 100):
                    #    wordVector[i] = wordVector[i]/len(words)

        printVector = list()
        if features[0] == 1:
            printVector += emotionVector
        if features[1] == 1:
            printVector += rulesVector
        if features[2] == 1:
            printVector += wordVector

        f.write(str(cls) + ' ' + ' '.join(['{}:{}'.format(x,y) for x,y in zip([i for i in range(1, len(printVector))], printVector)]) + '\r\n')
        #print('vector ' + str(count) + ' processed')
    f.close()

with open('/Users/ecnu/Desktop/ThesisResearch/implementations/testUnsegmented.csv', newline='') as f:
    testData = f.read().splitlines()
f.close()

for v in product([0,1], repeat=3):
    fVec = list(v)
    goldStandard = list()
    if sum(fVec) != 0:
        print(fVec)
        #create training vectors
        createVectors(trainData, 'train', fVec)
        #create the testing vectors
        createVectors(testData, 'test', fVec)

        #run the svm
        y, x = svm_read_problem('train1SVM.txt')
        print(len(x))
        print(len(y))
        m = svm_train(y, x, '-s 0 -t 0')
        y, x = svm_read_problem('test1SVM.txt')
        print(len(x))
        print(len(y))
        predictions, p_acc, p_val = svm_predict(y, x, m)

        #Evaluation
        target_names = ['fake', 'real']
        print(classification_report(goldStandard, predictions, target_names=target_names))
        errorVector = list()
        for i in range(0, len(goldStandard)):
            if goldStandard[i] != predictions[i]:
                errorVector.append(i)
        print('error vector')
        print(errorVector)
        print('ensemble output')
        print(predictions)
