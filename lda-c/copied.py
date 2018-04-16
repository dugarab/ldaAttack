import outer_optimization as outer
import matplotlib.pyplot as plt
from gensim import corpora, models, similarities,matutils
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import sparse as sp
import os
import time
def findVariationalParams(M,datapath,param,alpha,K):
    ''' 
    Function to determine the variational parameters
    Input: 
    M = DxV integer ndarray where 
           M(d,v) = Count of word v in document d
    alpha = K-vector of floats, shape = (K,) 
    beta = V-vector of floats, shape = (V,)
    
    Method: Use eq. 2,3,4 to determine outputs
    Refer to Blei et al.(2003) for definition of psi
    
    Output:
    eta: KxV float ndarray
    gamma: DxK float ndarray
    phi: DxVxK float ndarray 
         (returning a row_sparse D.V x K matrix phisp) 
    '''
    D,V = M.shape
    #K = alpha.shape[0]
    eta = np.ones ((K,V))
    gamma = np.ones((D,K))
    phi = np.ones((D,V,K))
    ################### CODE HERE #######################
    c_code = "lda-c/"
    cmd1 = "rm -r "+param
    # Ignore the os error that will come when no such file exists
    cmd2 = "mkdir "+param
    cmd3 = c_code+"lda est "+str(alpha)+" "+ str(K) +" "+ c_code + \
          "settings.txt " + datapath + " manual="+c_code+"ldaseeds.txt " + param + " >stdout.log 2>stderr.log"
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
    print("Reading phi")
    p = np.loadtxt(param+"/final.phi")
    coords = np.int32(p.T[0:-1,:])
    phi = sp.COO(coords,p[:,-1],shape=(D,V,K))
    print("Reading gamma")
    gamma = np.loadtxt(param+"/final.gamma")
    print("Reading eta")
    fee = (np.loadtxt(param+"/final.beta"))
    print("Estimating fee")
    eta = eta/(np.sum(eta,1).reshape(K,1)*0.1)
    return eta,gamma,phi,np.exp(fee)

def preprocessWords(inputPath,corpusfile,dcyfile,stop_words, nochange = True):
    '''
    Parses all files in the inputPath folder and
    returns the word matrix M:DxV of type ndarray(int32).
    Also stores the corpus in blei's LDA-C format as 
    corpusfile (corpusfile is a full path with filename).
    Input-specific stopwords also taken as array of strings
    '''
    porter = PorterStemmer()
    docs,docLen=[],0
    for path in inputPath:
        print("Reading data from %s"%path)
        for filename in os.listdir(path):
            with open(path+filename,'r') as inp:
                #print("Reading data from %s"%filename)
                f=inp.read()
                words=word_tokenize(f)
                words = [w.lower() for w in words]
                noPunc = [w.translate(None,string.punctuation)
                          for w in words]
                noEmp = [w for w in noPunc if w.isalpha()]
                noStop = [w for w in noEmp if not w
                          in stop_words]
                stemmed = [porter.stem(w) for w in noStop]
                stemmed = [w for w in stemmed if not w
                          in stop_words]
            docLen+=len(stemmed)
            docs.append(stemmed)
            #docs.append(noStop)
    D = len(docs)
    print ("Total Number of documents = %d"%D)
    print("Average words per document = %d"%(docLen/D))
    dcy = corpora.Dictionary(docs)
    #dcy['zzzzzz'] = 5793
    V = len(dcy)
    print("Total vocabulary size = %d"%V)
    dcy.save(dcyfile)
    corpus = [dcy.doc2bow(text) for text in docs]
    if not nochange:
        for doc in corpus:
            doc.append((V,1))
        V+=1

    corpora.BleiCorpus.serialize(corpusfile,corpus)
    M = matutils.corpus2dense(corpus, num_terms=V, num_docs=D,
                              dtype=np.int32).T
    return M

def runLDA(corpusfile,dcyfile,num_topics,ind = -1):
    '''
    Do classical LDA on word matrix M using alpha, beta
    Plot the results
    '''
    print("Running Vanilla LDA on current M")
    dcy = corpora.Dictionary.load(dcyfile)
    print(dcy)
    if ind >0 :
        tmp = dcy.token2id
        for key in tmp:
            if tmp[key] == int(ind):
                print ('Word to insert: ' + key)
                break
    
    corpus = corpora.BleiCorpus(corpusfile)
    #tfidf = models.TfidfModel(corpus, normalize=True)
    #tfidf_corpus = tfidf[corpus]
    tfidf_corpus = corpus  #Remove this line to allow tfidf values
    lda = models.LdaModel(tfidf_corpus, id2word=dcy, 
                          num_topics=num_topics)
    print(lda.print_topics(num_topics,num_words=20))
    return 0


def permuteFee(fee,fees_old):
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
corpFile2 = pth+"/congCorp2.lda-c"
dcyFile = pth+"/cong.dict"
paramFolder = pth +"/param" 
alpha = 0.1
K = 10
M_0 = preprocessWords(pathnames,corpFile,dcyFile,stop_words)
runLDA(corpFile,dcyFile,K)

M_0 = preprocessWords(pathnames,corpFile,dcyFile,stop_words)
#M_0 = np.concatenate((M_0, np.ones((M_0.shape[0],1),dtype = np.int32)), axis = 1)
#runLDA(corpFile,dcyFile,K)
eta,gamma,phi,fee=findVariationalParams(M_0,corpFile2,
                                        paramFolder,alpha,K)

D,V,K = phi.shape
print ("D=%d, V=%d, K=%d"%(D,V,K))
'''
We are poisoning our fee here to get the
poisoned feestar
'''

#######################################
feestar=np.copy(fee)

tmp1 = int(K/2)
tmp2 = np.argsort(fee[tmp1])[-500]

 

feestar[tmp1][tmp2]  = sorted(feestar[tmp1])[-5]

sum_row = np.sum(feestar[tmp1])

feestar[tmp1] = feestar[tmp1]/sum_row

########################################
eps = 0.005

M = np.copy(M_0)
momentum = 0
M_new = outer.update(eta, phi, fee,feestar, M_0, M,1,momentum,range(K))
it=1
a = []
a.append(0.5*np.linalg.norm(fee- feestar,'fro')**2)

print("Error: %f"%(a[-1]))
print ('Iteration %d complete'%it)


'''
    This is the main section where we optimize
    This is a bilevel optimization scheme, where we the inner
    optimisation (actual LDA) is done from blei's C code and we get eta, phi , gamma from there.
    Then outer optimisation minimises error between the fee obtained from the blei's LDA and our required feestar
'''
##while(0.5*np.linalg.norm((abs(fee-feestar) - eps)*(abs(fee-feestar) - eps>0),'fro')**2 > 0.0000001 and it<5):
##    M =(M_new)
##    #print("M-M_projected = %f"%(np.linalg.norm(M_new - np.float32(M)))) 
##    print('dimensions of M: %ix%i'%(M.shape[0], M.shape[1]))
##    corpus = matutils.Dense2Corpus(M,documents_columns=False)
##    corpora.BleiCorpus.serialize(corpFile,corpus)
##    eta,gamma,phi,fee=findVariationalParams(M,corpFile,paramFolder,alpha,K)
##    it+=1
##    M_new = outer.update(eta,phi,feestar,M_0,M,it)
##    print('Iteration %d complete'%it)
##    a.append(0.5*np.linalg.norm((abs(fee-feestar) - eps)*(abs(fee-feestar) - eps>0),'fro')**2)
##    print("norm(M-M_0): %.10f"%(np.linalg.norm(M_new-M_0,1)))
##    print("Error: %.10f"%(a[-1]))
##
##
##    
##M_final = outer.project_to_int(M[:,:-1])
###M_final = outer.project_to_int(M)
##corpus = matutils.Dense2Corpus(M_final,
##                               documents_columns=False)
##corpora.BleiCorpus.serialize(corpFile,corpus)
##runLDA(corpFile,dcyFile,K)
##t1=time.time()
##print ("Time taken = %f sec"%(t1-t0))
rankList = []
fee_old = np.copy(fee)
while(np.linalg.norm(fee-feestar) > 0.000001 and it<10):
    M =(M_new)
    #print("M-M_projected = %f"%(np.linalg.norm(M_new - np.float32(M)))) 
    print('dimensions of M: %ix%i'%(M.shape[0], M.shape[1]))
    corpus = matutils.Dense2Corpus(M,documents_columns=False)
    corpora.BleiCorpus.serialize(corpFile,corpus)
    eta,gamma,phi,fee=findVariationalParams(M,corpFile,paramFolder,alpha,K)
    feestar,perm = permuteFee(feestar,fee)
    
    a.append(0.5*np.linalg.norm(fee-feestar,'fro')**2)
    print("Error: %f"%(a[-1]))
    
    
    tmp_list  = sorted(fee[perm[tmp1]])
    
    word_val = fee[perm[tmp1]][tmp2]
    
    i = len(tmp_list)  - 1
    while(tmp_list[i] > word_val):
        #print(str(tmp_list[i]) + "  ," + str(word_val))
        i-=1
    rankList.append(V-i)
    print('Rank : ' + str(rankList[-1]))
    it+=1
    print('Iteration %d complete'%it)
   
    M_new = outer.update(eta,phi,fee,feestar,M_0,M,it,momentum,perm)
    print("norm(M-M_0): %f"%(np.linalg.norm(M_new-M_0,1)))
    


    
M_final = outer.project_to_int(M[:,:-1])
corpus = matutils.Dense2Corpus(M_final,
                               documents_columns=False)
corpora.BleiCorpus.serialize(corpFile,corpus)
runLDA(corpFile,dcyFile,K,tmp2)
t1=time.time()
print ("Time taken = %f sec"%(t1-t0))


plt.plot(range(len(a)),a, 'ro')
plt.plot(range(len(rankList)),rankList, '.')
#plt.axis([0, 6, 0, 20])
plt.show()
print(a)



