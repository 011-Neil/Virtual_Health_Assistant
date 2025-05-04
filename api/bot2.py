# # Meet Pybot: your friend
# import warnings
# warnings.filterwarnings("ignore")
# # nltk.download() # for downloading packages
# import numpy as np
# import random
# import string

# import nltk
# nltk.download('punkt_tab')
# # nltk.data.path.append('nltk_data')
# from collections import defaultdict

# from os.path import dirname, abspath, join

# # Get the absolute path to the current file's directory
# current_dir = dirname(abspath(__file__))

# # Use it for file paths
# nltk_data_path = join(current_dir, 'nltk_data')
# symptom_file = join(current_dir, 'symptom.txt')
# pincodes_file = join(current_dir, 'pincodes.txt')

# nltk.data.path.append(nltk_data_path)


# # File handling remains the same
# f=open('symptom.txt','r',errors = 'ignore')
# m=open('pincodes.txt','r',errors = 'ignore')
# raw=f.read()
# rawone=m.read()
# raw=raw.lower()
# rawone=rawone.lower()
# sent_tokens = nltk.sent_tokenize(raw)
# word_tokens = nltk.word_tokenize(raw)
# sent_tokensone = nltk.sent_tokenize(rawone)
# word_tokensone = nltk.word_tokenize(rawone)

# # Keep original lemmatization setup
# lemmer = nltk.stem.WordNetLemmatizer()
# def LemTokens(tokens):
#     return [lemmer.lemmatize(token) for token in tokens]
# remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
# def LemNormalize(text):
#     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# # Original conversation patterns
# Introduce_Ans = [" "]
# GREETING_INPUTS = ("hello", "hi","hiii","hii","hiiii","hiiii", "greetings", "sup", "what's up","hey",)
# GREETING_RESPONSES = ["hi,are you suffering from any health issues?(Y/N)", "hey,are you having any health issues?(Y/N)", "hii there,are you having any health issues?(Y/N)", "hi there,are you having any health issues?(Y/N)", "hello,are you having any health issues?(Y/N)", "I am glad! You are talking to me,are you having any health issues?(Y/N)"]
# Basic_Q = ("yes","y")
# Basic_Ans = "okay,tell me about your symptoms"
# Basic_Om = ("no","n")
# Basic_AnsM = "thank you visit again"
# fev=("iam suffering from fever", "i affected with fever","i have fever","fever")
# feve_r=("which type of fever you have? and please mention your symptoms then we try to calculate your disease.")

# # Original greeting functions
# def greeting(sentence):
#     for word in sentence.split():
#         if word.lower() in GREETING_INPUTS:
#             return random.choice(GREETING_RESPONSES)

# def basic(sentence):
#     for word in Basic_Q:
#         if sentence.lower() == word:
#             return Basic_Ans

# def fever(sentence):
#     for word in fev:
#         if sentence.lower() == word:
#             return feve_r

# def basicM(sentence):
#     for word in Basic_Om:
#         if sentence.lower() == word:
#             return Basic_AnsM

# def IntroduceMe(sentence):
#     return random.choice(Introduce_Ans)

# # Modified TF-IDF implementation using numpy only
# def compute_tfidf(documents):
#     """Manual TF-IDF implementation using numpy"""
#     tokenized_docs = [LemNormalize(doc) for doc in documents]
#     idf = defaultdict(float)  # Now properly imported
#     doc_count = len(documents)

#     # Rest of the function remains unchanged
#     for tokens in tokenized_docs:
#         for word in set(tokens):
#             idf[word] += 1

#     # Smooth IDF
#     for word in idf:
#         idf[word] = np.log(doc_count / (1 + idf[word])) + 1

#     # Create vocabulary
#     vocab = list(idf.keys())
#     word_to_idx = {word: i for i, word in enumerate(vocab)}

#     # Calculate TF-IDF vectors
#     tfidf_vectors = []
#     for tokens in tokenized_docs:
#         tf = defaultdict(float)
#         for word in tokens:
#             tf[word] += 1
#         doc_len = len(tokens)
#         vector = np.zeros(len(vocab))
#         for word, count in tf.items():
#             if word in word_to_idx:
#                 vector[word_to_idx[word]] = (count/doc_len) * idf[word]
#         tfidf_vectors.append(vector)
    
#     return np.array(tfidf_vectors), vocab

# def cosine_similarity(matrix):
#     """Cosine similarity implementation using numpy"""
#     norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#     norm_matrix = matrix / norms
#     return np.dot(norm_matrix, norm_matrix.T)

# # Modified response functions with numpy
# def response(user_response):
#     robo_response = ''
#     sent_tokens.append(user_response)
#     try:
#         tfidf_matrix, _ = compute_tfidf(sent_tokens)
#         similarity_matrix = cosine_similarity(tfidf_matrix)
#         user_sim = similarity_matrix[-1][:-1]  # Exclude self
        
#         if len(user_sim) == 0 or (max_sim := np.max(user_sim)) == 0:
#             robo_response = "I am sorry! I don't understand you"
#         else:
#             idx = np.argmax(user_sim)
#             robo_response = sent_tokens[idx]
#     finally:
#         sent_tokens.pop()
#     return robo_response

# def responseone(user_response):
#     robo_response = ''
#     sent_tokensone.append(user_response)
#     try:
#         tfidf_matrix, _ = compute_tfidf(sent_tokensone)
#         similarity_matrix = cosine_similarity(tfidf_matrix)
#         user_sim = similarity_matrix[-1][:-1]
        
#         if len(user_sim) == 0 or (max_sim := np.max(user_sim)) == 0:
#             robo_response = "I am sorry! I don't understand you"
#         else:
#             idx = np.argmax(user_sim)
#             robo_response = sent_tokensone[idx]
#     finally:
#         sent_tokensone.pop()
#     return robo_response

# # Original chat function remains unchanged
# def chat(user_response):
#     user_response=user_response.lower()
#     keyword = " module "
#     keywordone = " module"
#     keywordsecond = "module "

#     if(user_response!='bye'):
#         if(user_response=='thanks' or user_response=='thank you' ):
#             return "You are welcome.."
#         elif(basicM(user_response)!=None):
#             return basicM(user_response)
#         else:
#             if(user_response.find(keyword) != -1 or user_response.find(keywordone) != -1 or user_response.find(keywordsecond) != -1):
#                 return responseone(user_response)
#             elif(greeting(user_response)!=None):
#                 return greeting(user_response)
#             elif(user_response.find("your name") != -1 or user_response.find(" your name") != -1 or user_response.find("your name ") != -1 or user_response.find(" your name ") != -1):
#                 return IntroduceMe(user_response)
#             elif(basic(user_response)!=None):
#                 return basic(user_response)
#             elif(fever(user_response)!=None):
#                 return fever(user_response)
#             else:
#                 return response(user_response)
#     else:
#         return "Bye! take care.."



import warnings
warnings.filterwarnings("ignore")
import random
import string
from groq import Groq
import pandas as pd 
import os

apiKey = os.environ.get('GROQ_API_KEY')
symptom_precaution = pd.read_csv('symptom_precaution.csv')
Symptom_severity =  pd.read_csv('Symptom-severity.csv')
symptom_Description = pd.read_csv('symptom_Description.csv')

# Initialize the Groq client
client = Groq(api_key='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')

# Define the system prompt for LLaMA
prompt = '''
You are a highly intelligent and medically-aware Virtual Health Assistant built to help users understand their health conditions based on symptoms they describe. You are not a licensed doctor but are designed to provide accurate, well-informed, and empathetic suggestions.

Your responsibilities are:

1. 🔍 Symptom Collection:
- Begin by asking the user to describe their current symptoms in natural language.
- Extract relevant symptoms by matching against a predefined symptom list.
- Be patient and conversational. If symptoms are vague or missing, gently ask follow-up questions to clarify.

2. 🧠 Disease Prediction:
- Use the {symptom_Description} dataset to analyze the user's symptoms and predict the most likely disease.
- Prioritize predictions with a confidence level of at least 90%. If multiple conditions are possible, list the top 1–2 likely conditions with confidence scores.
- Provide a short, clear medical explanation of the predicted condition using data from the dataset.

3. ✅ Precaution Guidance (on request or when appropriate):
- If the user asks for precautions or if the disease is moderately/severely serious, use the {symptom_precaution} dataset.
- Present 3–4 easy-to-follow precautions in bullet-point format to help the user manage their condition safely.

4. 🚨 Severity Analysis:
- Use the {Symptom_severity} dataset to calculate a total severity score based on the user's listed symptoms.
- Classify severity into:
  - 🟢 Mild (score < 5)
  - 🟠 Moderate (5–10)
  - 🔴 Severe (score > 10)
- Explain what the severity means, and whether the user should rest, monitor, or immediately consult a doctor.

5. 👨‍⚕️ Doctor Recommendation:
- Based on the predicted disease, recommend the appropriate specialist (e.g., Dermatologist, Neurologist, General Physician).
- If the user provides a pincode, use it to search a doctor database (or mocked doctor directory) and return:
  - Doctor’s Name
  - Specialization
  - Clinic/Hospital Name
  - Contact Details

6. ❓ Follow-up Support:
- After responding, remain ready for any follow-up questions the user might have.
- Handle queries like:
  - “What should I do next?”
  - “Is this contagious?”
  - “Can I treat this at home?”
  - “Can you suggest medicine?”
- Always provide safe, general advice and encourage users to consult real doctors for serious or persistent symptoms.

Be polite, friendly, and professional at all times. Your primary goals are user safety, accuracy of prediction, and clear communication your name Med.
'''


# Define the chat function that uses LLaMA model
def query_virtual_health_assistant(user_input):
    response_text = ""

    # Interact with Groq's API using LLaMA-3 model
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=True,
        stop=None,
    )

    # Collect the response from the model
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            response_text += content

    return response_text.strip()

# Simplified chat function, without the TF-IDF logic
def chat(user_response):
    user_response = user_response.lower()

    # Basic greeting logic
    GREETING_INPUTS = ("hello", "hi", "hiii", "hey", "sup")
    GREETING_RESPONSES = ["Hi there! Are you experiencing any health issues today? (Y/N)"]

    if user_response in GREETING_INPUTS:
        return random.choice(GREETING_RESPONSES)

    # Handle user response for health issues
    if user_response in ("yes", "y"):
        return "Okay, tell me about your symptoms, and I will try to help."
    
    elif user_response in ("no", "n"):
        return "Thank you for visiting. Take care!"

    # Otherwise, forward the symptoms to the LLaMA model
    if user_response != 'bye':
        return query_virtual_health_assistant(user_response)

    else:
        return "Goodbye! Take care of your health."

# # Example usage
# while True:
#     user_input = input("You: ")
#     response = chat(user_input)
#     print("Pybot: ", response)
#     if user_input.lower() == 'bye':
#         break
