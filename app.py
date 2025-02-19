from flask import Flask, request, jsonify, render_template
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)


# Load and read the data
f = open('./Data.txt', 'r', errors='ignore')
raw_doc = f.read()

# Preprocess the document
raw_doc = raw_doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

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

# Route to handle the chatbot conversation
@app.route("/chat", methods=['POST'])
def chat():
    try:
        user_response = request.json.get("message")  # Get user input from the JSON request
        if user_response:
            if greet(user_response) is not None:
                return jsonify({"response": greet(user_response)})
            else:
                sentence_tokens.append(user_response)  # Add the user input to sentence tokens
                word_tokens.extend(nltk.word_tokenize(user_response))  # Add the tokens of the response
                
                # Remove duplicates
                final_words = list(set(word_tokens))
                
                bot_response = response(user_response)
                
                sentence_tokens.remove(user_response)  # Remove the user input after processing
                return jsonify({"response": bot_response})
        else:
            return jsonify({"response": "Please send a valid message."})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# Route for the homepage (UI)
@app.route("/")
def home():
    return render_template("index.html")  # Renders the HTML file

if __name__ == "__main__":
    app.run(debug=True)
