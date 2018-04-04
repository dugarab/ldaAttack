from gensim import corpora, models, similarities,matutils
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import scipy.sparse as sp
import os
import time
import outer_optimization as outer
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
          "settings.txt " + datapath + " random " + param
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
    print("Reading phi")
    phi = np.loadtxt(param+"/final.phi").reshape(D,V,K)
    print("Reading gamma")
    gamma = np.loadtxt(param+"/final.gamma")
    print("Reading beta")
    beta = np.loadtxt(param+"/final.beta")
    print ("Building eta")
    for v in xrange(V):
        s=np.zeros((K,))
        for d in xrange(D):
            s+=M[d,v]*phi.T[:,v,d]
        eta[:,v] = beta[:,v] + s
    phisp = sp.csr_matrix(phi.reshape(D*V,K))
    #Use phisp preferably for operations
    #Usage phi[d,v,k]=phisp[d*V+v,k]
    return eta,gamma,phisp,beta


def preprocessWords(inputPath,corpusfile,stop_words):
    '''
    Parses all files in the inputPath folder and
    returns the word matrix M:DxV of type ndarray(int32).
    Also stores the corpus in blei's LDA-C format as 
    corpusfile (corpusfile is a full path with filename).
    Input-specific stopwords also taken as array of strings
    '''
    porter = PorterStemmer()
    docs,docLen=[],0
    print("Reading data from %s"%inputPath)
    for path in inputPath:
        for filename in os.listdir(path):
            with open(path+filename,'r') as inp:
                print("Reading data from %s"%filename)
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
    V = len(dcy)
    print("Total vocabulary size = %d"%V)
    #dcy.save(os.path.join(TMP,'cong.dict'))
    corpus = [dcy.doc2bow(text) for text in docs]
    corpora.BleiCorpus.serialize(corpusfile,corpus)
    M = matutils.corpus2dense(corpus, num_terms=V, num_docs=D,
                              dtype=np.int32).T
    return M

def runLDA(M,alpha,beta):
    '''
    Do classical LDA on word matrix M using alpha, beta
    Plot the results
    '''
    return 0


if __name__ =="__main__":
    '''
    Write the main function
    '''
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
    pth = "/Users/ishaan/MLPdatafiles"
    # Create a path where you want to keep your output files
    os.system("rm -r "+pth)
    # Ignore the os error that will come when no such file exists
    os.system("mkdir "+pth)
    corpFile = pth+"/congCorp.lda-c"
    paramFolder = pth +"/param" 
    alpha = 0.1
    K = 10
    M = preprocessWords(pathnames,corpFile,stop_words)
    eta,gamma,phisp,beta=findVariationalParams(M,corpFile,
                                               paramFolder,alpha,K)
    t1=time.time()
    print ("Time taken = %f sec"%(t1-t0))
    

