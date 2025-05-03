 # Meet Pybot: your friend
import warnings
warnings.filterwarnings("ignore")
# nltk.download() # for downloading packages
#import tensorflow as tf
import numpy as np
import random
import string # to process standard python strings
import nltk
nltk.data.path.append('nltk_data')




f=open('symptom.txt','r',errors = 'ignore')
m=open('pincodes.txt','r',errors = 'ignore')
checkpoint = "./chatbot_weights.ckpt"
#session = tf.InteractiveSession()
#session.run(tf.global_variables_initializer())
#saver = tf.train.Saver()
#saver.restore(session, checkpoint)

raw=f.read()
rawone=m.read()
raw=raw.lower()# converts to lowercase
rawone=rawone.lower()# converts to lowercase
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words
sent_tokensone = nltk.sent_tokenize(rawone)# converts to list of sentences
word_tokensone = nltk.word_tokenize(rawone)# converts to list of words


sent_tokens[:2]
sent_tokensone[:2]

word_tokens[:5]
word_tokensone[:5]

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

Introduce_Ans = [" "]
GREETING_INPUTS = ("hello", "hi","hiii","hii","hiiii","hiiii", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi,are you suffering from any health issues?(Y/N)", "hey,are you having any health issues?(Y/N)", "hii there,are you having any health issues?(Y/N)", "hi there,are you having any health issues?(Y/N)", "hello,are you having any health issues?(Y/N)", "I am glad! You are talking to me,are you having any health issues?(Y/N)"]
Basic_Q = ("yes","y")
Basic_Ans = "okay,tell me about your symptoms"
Basic_Om = ("no","n")
Basic_AnsM = "thank you visit again"
fev=("iam suffering from fever", "i affected with fever","i have fever","fever")
feve_r=("which type of fever you have? and please mention your symptoms then we try to calculate your disease.")


# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Checking for Basic_Q
def basic(sentence):
    for word in Basic_Q:
        if sentence.lower() == word:
            return Basic_Ans
def fever(sentence):
    for word in fev:
        if sentence.lower() == word:
            return feve_r

# Checking for Basic_QM
def basicM(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Om:
        if sentence.lower() == word:


            return Basic_AnsM
# Checking for Introduce
def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)

# ... (keep previous imports but REMOVE sklearn imports)
from sparse import COO, dot, stack as sparse_stack
from collections import defaultdict

# Add these functions
def compute_tfidf(documents):
    """Manual TF-IDF implementation using sparse"""
    tokenized_docs = [LemNormalize(doc) for doc in documents]
    idf = defaultdict(float)
    doc_count = len(documents)

    # Calculate IDF
    for tokens in tokenized_docs:
        for word in set(tokens):
            idf[word] += 1

    # Smooth IDF
    for word in idf:
        idf[word] = np.log(doc_count / (1 + idf[word])) + 1

    # Create vocabulary
    vocab = list(idf.keys())
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    # Calculate TF-IDF vectors
    tfidf_vectors = []
    for tokens in tokenized_docs:
        tf = defaultdict(float)
        for word in tokens:
            tf[word] += 1
        doc_len = len(tokens)
        vector = np.zeros(len(vocab))
        for word, count in tf.items():
            if word in word_to_idx:
                vector[word_to_idx[word]] = (count/doc_len) * idf[word]
        tfidf_vectors.append(COO.from_numpy(vector))
    
    return sparse_stack(tfidf_vectors), vocab

def sparse_cosine_similarity(matrix):
    """Cosine similarity for sparse matrices"""
    norms = np.sqrt((matrix * matrix).sum(axis=1))
    norm_matrix = matrix / norms[:, None]
    return norm_matrix @ norm_matrix.T

# Modified response functions
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    try:
        tfidf_matrix, _ = compute_tfidf(sent_tokens)
        similarity_matrix = sparse_cosine_similarity(tfidf_matrix)
        user_sim = similarity_matrix[-1].todense().A1[:-1]  # Exclude self
        
        if len(user_sim) == 0 or (max_sim := np.max(user_sim)) == 0:
            robo_response = "I am sorry! I don't understand you"
        else:
            idx = np.argmax(user_sim)
            robo_response = sent_tokens[idx]
    finally:
        sent_tokens.pop()
    return robo_response

def responseone(user_response):
    robo_response = ''
    sent_tokensone.append(user_response)
    try:
        tfidf_matrix, _ = compute_tfidf(sent_tokensone)
        similarity_matrix = sparse_cosine_similarity(tfidf_matrix)
        user_sim = similarity_matrix[-1].todense().A1[:-1]
        
        if len(user_sim) == 0 or (max_sim := np.max(user_sim)) == 0:
            robo_response = "I am sorry! I don't understand you"
        else:
            idx = np.argmax(user_sim)
            robo_response = sent_tokensone[idx]
    finally:
        sent_tokensone.pop()
    return robo_response

# Keep the rest of your code unchanged (chat function, etc.)

def chat(user_response):
    user_response=user_response.lower()
    keyword = " module "
    keywordone = " module"
    keywordsecond = "module "

    if(user_response!='bye'):

        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            #print("ROBO: You are welcome..")
            return "You are welcome.."
        elif(basicM(user_response)!=None):
            return basicM(user_response)
        else:
            if(user_response.find(keyword) != -1 or user_response.find(keywordone) != -1 or user_response.find(keywordsecond) != -1):
                #print("ROBO: ",end="")
                #print(responseone(user_response))
                return responseone(user_response)
                sent_tokensone.remove(user_response)
            elif(greeting(user_response)!=None):
                #print("ROBO: "+greeting(user_response))
                return greeting(user_response)
            elif(user_response.find("your name") != -1 or user_response.find(" your name") != -1 or user_response.find("your name ") != -1 or user_response.find(" your name ") != -1):
                return IntroduceMe(user_response)
            elif(basic(user_response)!=None):
                return basic(user_response)
            elif(fever(user_response)!=None):
                return fever(user_response)
            else:
                #print("ROBO: ",end="")
                #print(response(user_response))
                return response(user_response)
                sent_tokens.remove(user_response)

    else:
        flag=False
        #print("ROBO: Bye! take care..")
        return "Bye! take care.."
