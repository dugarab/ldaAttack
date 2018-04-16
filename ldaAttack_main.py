import outer_optimization as outer
import vanillaLDA as vlda 
import matplotlib.pyplot as plt
from gensim import corpora, models, similarities,matutils
import string
from nltk.corpus import stopwords
import numpy as np
import sparse as sp
import os
import time

def findVariationalParams(datapath,param,D,V,alpha=0.1,K=10):
    ''' 
    Function to determine the variational parameters
    Input: 
    M = DxV integer ndarray where 
           M(d,v) = Count of word v in document d
    alpha = Real Parameter alpha
    
    Output:
    eta: KxV float ndarray
    gamma: DxK float ndarray
    phi: DxVxK float sparse ndarray 
    '''
    eta = np.ones ((K,V))
    gamma = np.ones((D,K))
    phi = np.ones((D,V,K))
    vlda.runBleiLDA(datapath,param,alpha,K)
    print("Reading Variational Parameters")
    p = np.loadtxt(param+"/final.phi")
    coords = np.int32(p.T[0:-1,:])
    phi = sp.COO(coords,p[:,-1],shape=(D,V,K))
    gamma = np.loadtxt(param+"/final.gamma")
    fee = np.exp(np.loadtxt(param+"/final.beta"))
    #eta = np.loadtxt(param+"/final.eta")
    #fee = eta/(np.sum(eta,1).reshape(K,1)*0.1)
    print ("Reading Done")
    return eta,gamma,phi,fee


def permuteFee(fee,fee_old):
    K,V = fee.shape
    feeperm = np.zeros((K,V))
    perm = [0]*K
    arr=[]
    for i in xrange(K):
        indices=np.argsort(np.linalg.norm(fee-fee_old[i],1,1))
        for j in indices:
            if j not in arr:
                feeperm[i]=fee[j]
                perm[i] = j
                arr.append(j)
                break
    return feeperm,perm

def loss(fee,feestar,num=1):
   eps = 0.005
   if num == 1:
       return 0.5*np.linalg.norm(fee-feestar,'fro')**2
   else:
       return 0.5*np.linalg.norm((abs(fee-feestar) - eps)*(abs(fee-feestar) - eps>0),'fro')**2
   
def rankOf(key,arr):
    '''
    Returns rank of arr[key] in arr
    '''
    val = arr[key]
    arrSort = sorted(arr)
    v = len(arr)
    for i in xrange(v-1,-1,-1):
        if arrSort[i]<=val:
            return v-i
    return 0

def generateTargetFee(fee,topic_ID,word_ID,dcyFile):
    '''
    We are poisoning our fee here to get the
    poisoned feestar
    '''
    feestar=np.copy(fee)
    tmp1 = topic_ID
    tmp2 = word_ID
    feestar[tmp1][tmp2]  = sorted(feestar[tmp1])[-5]
    sum_row = np.sum(feestar[tmp1])
    feestar[tmp1] = feestar[tmp1]/sum_row
    dcy = corpora.Dictionary.load(dcyFile)
    tmp = dcy.token2id
    for key in tmp:
        if tmp[key] == tmp2:
            w2i = key
            break
    return feestar,w2i

if __name__=='__main__':
    t0=time.time()
    stop_words = stopwords.words('english')
    stop_words += ['mr','would','say','lt', 'p', 'gt',
                   'amp', 'nbsp','bill','speaker','us',
                   'going','act','gentleman','gentlewoman',
                   'chairman','nay','yea','thank']
    pathnames = ['./convote_v1.1/data_stage_one/'+wor+'/'
                 for wor in ['development_set']]#,'training_set']]
    # Use development test(702 docs) only for debugging
    # i.e. Remove 'training set' from wor in pathnames
    pth = "MLPdatafiles"
    # Create a path where you want to keep your output files
    os.system("rm -r "+pth)
    # Ignore the os error that will come when no such file exists
    os.system("mkdir "+pth)
    corpFile = pth+"/congCorp.lda-c"
    vocabFile = corpFile+".vocab"
    dcyFile = pth+"/cong.dict"
    paramFolder = pth +"/param"
    betaFile = paramFolder+"/final.beta"
    init_out = pth+"/bleiOut.init.dat"
    final_out = pth+"/bleiOut.final.dat"
    alpha = 0.1
    K = 10
    M_0 = vlda.preprocessWords(pathnames,corpFile,dcyFile,stop_words)
    D,V = M_0.shape
    eta,gamma,phi,fee=findVariationalParams(corpFile,paramFolder,D,V,alpha,K)
    vlda.print_topics(betaFile,vocabFile,dcyFile,init_out,nwords=20)
    ############### Start attacking the corpus ######################
    eps = 0.005
    M = np.copy(M_0)
    momentum = 0
    TOL = 0.0000001
    maxiter = 20
    it=1
    riskNorm,wordRank = [],[]
    #Inserting word with 500th rank in top 10 in Topic K/2
    topicID,wordID = K/2,np.argsort(fee[K/2])[-500]
    feestar,w2i = generateTargetFee(fee,topicID,wordID,dcyFile)
    print("Attacking corpus to insert \'%s\' in topic no. %d\n"%(w2i,topicID))
    M_new = outer.update(eta, phi, fee,feestar, M_0, M,1,momentum)
    riskNorm.append(loss(fee,feestar,0))
    wordRank.append(rankOf(wordID,fee[topicID]))
    print("Rank of word \'%s\' = %d"%(w2i,wordRank[-1]))
    print("Risk function ||fee-feestar||: %f"%(riskNorm[-1]))
    print ('Iteration %d complete'%it)
    print ('************************************************************\n')
    while(np.linalg.norm(fee-feestar) > TOL and it<maxiter):
        M=M_new
        print('dimensions of M: %ix%i'%(M.shape[0], M.shape[1]))
        corpus = matutils.Dense2Corpus(M,documents_columns=False)
        corpora.BleiCorpus.serialize(corpFile,corpus)
        eta,gamma,phi,fee=findVariationalParams(corpFile,paramFolder,D,V,alpha,K)
        feestar,perm = permuteFee(feestar,fee)
        riskNorm.append(loss(fee,feestar,0))
        M_new= outer.update(eta,phi,fee,feestar,M_0,M,it,momentum)
        wordRank.append(rankOf(wordID,fee[perm[topicID]]))
        it+=1
        print("norm(M-M_0): %f"%(np.linalg.norm(M_new-M_0,1)))
        print("Risk Function ||fee-feestar||: %f"%(riskNorm[-1]))
        print("Rank of word \'%s\' = %d"%(w2i,wordRank[-1]))
        print('Iteration %d complete'%it)
        print ('************************************************************\n')
    M_final = np.int32(M)
    corpus = matutils.Dense2Corpus(M_final,documents_columns=False)
    corpora.BleiCorpus.serialize(corpFile,corpus)
    vlda.runBleiLDA(corpFile,paramFolder,alpha,K)
    vlda.print_topics(betaFile,vocabFile,dcyFile,final_out,nwords=20)
    t1=time.time()
    print ("Net Time taken = %f sec"%(t1-t0))
    plt.plot(range(1,len(riskNorm)+1),riskNorm, 'r')
    plt.plot(range(1,len(rnk)+1),rnk,'b')
    # plt.title("Loss function value with iterations")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss function norm")
    plt.show()


