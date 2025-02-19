import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and read the data
f = open('C:/Workspace/Portfolio/Project/Data.txt', 'r', errors='ignore')
raw_doc = f.read()

# Preprocess the document
raw_doc = raw_doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

print(sentence_tokens[:5])
# Lemmatization setup
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def lemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Greetings setup
Greet_inputs = (
    "hello", "hi", "hey", "hii", "howdy", "greetings", "what's up", "salutations", 
    "good morning", "good afternoon", "yo", "hi there", "hey there", "what's going on", 
    "good evening", "how's it going", "hey hey", "what's up doc", "how's everything", "how's life"
)

Greet_responses = (
    "Hi, Nice to meet you. How can I help you?", "Hello, Nice to meet you. How can I help you?", 
    "Hey, Nice to meet you. How can I help you?", "Hi there, how can I assist you today?", 
    "Greetings, how may I assist you?", "Howdy, what can I do for you?", 
    "What's up! How can I help you today?", "Salutations! How can I assist you?", 
    "Good morning! How can I be of help?", "Good afternoon! How may I help?", 
    "Yo, how can I assist you?", "Hi there! What can I help with today?", 
    "Hey there! What can I do for you?", "What's going on? How can I assist you?", 
    "Good evening! How may I assist you?", "How's it going? What can I help you with?", 
    "Hey hey! How can I help you today?", "What's up, doc? How can I assist you?", 
    "How's everything? How may I help?", "How's life? What can I do for you?"
)


def greet(sentence):
    for word in sentence.split():
        if word.lower() in Greet_inputs:
            return random.choice(Greet_responses)

# Response generation function
def response(user_response):
    robol_response = ''
    TfidVec = TfidfVectorizer(tokenizer=lemNormalize, stop_words='english')
    tfidf = TfidVec.fit_transform(sentence_tokens)
    
    # Transform the user's response into the same vector space
    user_input_tfidf = TfidVec.transform([user_response])
    
    # Calculate cosine similarity
    vals = cosine_similarity(user_input_tfidf, tfidf)
    
    # Sort similarities and get the most similar sentence
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robol_response = "I am sorry. Unable to understand you!"
    else:
        robol_response = sentence_tokens[idx]
    
    return robol_response

# Main interaction loop
flag = True
print("Hello! I am the AI Assistant of NPPD. Start typing your text after greeting to talk to me. Type 'bye' to end.")
while flag:
    user_response = input("You: ")
    user_response = user_response.lower()
    
    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            flag = False
            print('NPPD: You are welcome!')
        else:
            if greet(user_response) != None:
                print('NPPD: ' + greet(user_response))
            else:
                sentence_tokens.append(user_response)  # Add the user input to sentence tokens
                word_tokens.extend(nltk.word_tokenize(user_response))  # Add the tokens of the response
                
                # Remove duplicates
                final_words = list(set(word_tokens))
                
                print("NPPD: ", end="")
                print(response(user_response))
                
                sentence_tokens.remove(user_response)  # Remove the user input after processing
                
    else:
        flag = False
        print("NPPD: Goodbye!")