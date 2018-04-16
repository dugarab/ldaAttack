from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities,matutils
import numpy as np
import os
import string
def preprocessWords(inputPath,corpusfile,dcyfile,stop_words):
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
    D = len(docs)
    print ("Total Number of documents = %d"%D)
    print("Average words per document = %d"%(docLen/D))
    dcy = corpora.Dictionary(docs)
    V = len(dcy)
    print("Total vocabulary size = %d"%V)
    print("Dictionary saved in "+dcyfile)
    dcy.save(dcyfile)
    corpus = [dcy.doc2bow(text) for text in docs]
    corpora.BleiCorpus.serialize(corpusfile,corpus)
    print("Corpus saved in "+corpusfile)
    M = matutils.corpus2dense(corpus, num_terms=V, num_docs=D,
                              dtype=np.int32).T
    return M

def runBleiLDA(datapath,param,alpha=0.1,K=10):
    c_code = "lda-c/"
    cmd1 = "rm -r "+param
    # Ignore the os error that will come when no such file exists
    cmd2 = "mkdir "+param
    cmd3 = c_code+"lda est "+str(alpha)+" "+ str(K) +" "+ c_code + \
          "settings.txt " + datapath + " manual="+c_code+"ldaseeds.txt " + param + " >stdout.log 2>stderr.log"
    os.system(cmd1)
    os.system(cmd2)
    print ("Running Blei's LDA code")
    os.system(cmd3)
    print ("Run completed")

def print_topics(beta_file, vocab_file, dcyfile, outfile="ldaTopics.txt", nwords=25):
    # get the vocabulary
    vocab = file(vocab_file, 'r').readlines()
    vocab = map(lambda x: x.strip(), vocab)
    # for each line in the beta file
    indices = range(len(vocab))
    topic_no = 0
    with open(outfile,'w') as f:
        for topic in file(beta_file, 'r'):
            f.write('topic %03d' % topic_no+'\n')
            topic = map(float, topic.split())
            indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
            for i in range(nwords):
                f.write( '   %s' % vocab[indices[i]]+'\n')
            topic_no = topic_no + 1
            f.write('\n')

    
def runGensimLDA(corpusfile,dcyfile,outfile,num_topics=10,nwords=25,tfidf=False):
    print("Running Gensim LDA on current M")
    dcy = corpora.Dictionary.load(dcyfile)
    corpus = corpora.BleiCorpus(corpusfile)
    if tfidf:
        tfidf = models.TfidfModel(corpus, normalize=True)
        corpus = tfidf[corpus]
    lda = models.LdaModel(corpus, id2word=dcy, 
                          num_topics=num_topics)
    with open(outfile,'w') as out:
        out.write(lda.print_topics(num_topics,nwords))
    return 0


