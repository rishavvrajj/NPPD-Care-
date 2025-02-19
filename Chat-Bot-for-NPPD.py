import random
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Sample chatbot data (FAQs)
responses = {
    "booking": [
        "You can book an appointment on our website.",
        "Visit our booking page to schedule your service.",
        "Booking is easy! Just follow the instructions on our website.",
        "To book an appointment, go to our booking section and select a time.",
        "Simply choose a date and time that works for you to book your appointment.",
        "You can schedule your service online. Just visit the booking page.",
        "Need help booking an appointment? Let me know!",
        "Head to our booking page to secure a time for your appointment.",
        "You can reserve your spot through our online booking system.",
        "Our website allows you to easily schedule an appointment in minutes."
    ],
    "refund": [
        "Refunds take 5-7 business days.",
        "Please provide your order ID for a refund request.",
        "We process refunds as quickly as possible. Please share your order ID.",
        "To initiate a refund, please contact our support team with your order details.",
        "Refund requests are handled within 7 days of receiving the product back.",
        "If you're eligible for a refund, it will be credited within 5 business days.",
        "For refund inquiries, send us your order number and we'll process it promptly.",
        "You can request a refund by providing the reason for your return and order ID.",
        "Refunds are typically issued to the original payment method.",
        "If you've received a damaged product, please contact us for a faster refund process."
    ],
    "support": [
        "Our support team is available 24/7.",
        "How can I assist you with your issue?",
        "If you need help, feel free to reach out to our customer support team anytime.",
        "Our support team is here to help. What seems to be the problem?",
        "Have a question or issue? Our support team is ready to assist.",
        "If you're experiencing a technical problem, our support team can help resolve it.",
        "You can contact support by sending us an email or through our support page.",
        "We offer 24/7 support, feel free to contact us anytime.",
        "Our support is always available to help you with any challenges.",
        "For urgent issues, our live support chat is the best option to reach out."
    ],
    "pricing": [
        "Pricing details are available on our website.",
        "We offer various pricing plans. Check our website for more info.",
        "Our pricing depends on the plan you choose. You can find the details on our pricing page.",
        "The cost varies based on the service you need. Visit our pricing page for more details.",
        "You can compare our pricing options on the pricing section of our website.",
        "For more information on prices, visit our pricing guide on the website.",
        "We offer discounts for long-term plans. Check out our special offers on the pricing page.",
        "Pricing varies depending on the package you select. More details are available on our website.",
        "For a detailed breakdown of pricing, please refer to our pricing FAQ section.",
        "Our pricing is competitive, and we offer flexible payment options."
    ],
    "shipping": [
        "We ship worldwide! Delivery times depend on your location.",
        "Shipping usually takes 3-5 business days depending on your location.",
        "Our shipping is fast and reliable! Check your order status anytime through our website.",
        "We offer standard and expedited shipping options during checkout.",
        "Shipping costs are calculated based on your delivery location.",
        "You can track your shipment using the tracking ID provided in the confirmation email.",
        "If you are in a hurry, we offer express shipping at checkout.",
        "We offer free shipping on orders over $50. Check our shipping policy for more details.",
        "Shipping typically takes 5-7 days for international orders.",
        "For urgent deliveries, we offer same-day shipping on select products."
    ],
    "account": [
        "You can create an account by visiting our website and clicking 'Sign Up'.",
        "If you have an existing account, you can log in using your email and password.",
        "Having trouble logging in? Click 'Forgot Password' to reset it.",
        "To manage your account, simply visit the account settings page.",
        "You can update your personal information and preferences from your account settings.",
        "To create a new account, just enter your email, password, and basic information.",
        "You can delete your account anytime by contacting our support team.",
        "To link your social media to your account, go to the account settings page.",
        "If you've forgotten your username, we can help you recover it.",
        "Your account settings can be accessed directly from the main dashboard after logging in."
    ],
    "payment": [
        "We accept all major credit cards and PayPal.",
        "Payments are securely processed through our payment gateway.",
        "For subscriptions, payments are charged monthly/annually depending on your plan.",
        "You can save your payment details for faster future purchases.",
        "We offer multiple payment methods including credit cards, PayPal, and more.",
        "Your payment information is encrypted and securely handled during checkout.",
        "If you face any issues with payment, feel free to reach out to support.",
        "We offer a 30-day money-back guarantee with every purchase.",
        "For any payment-related inquiries, please contact our support team.",
        "You can easily update your payment information from the billing section of your account."
    ],
    "cancellation": [
        "You can cancel your order within 24 hours for a full refund.",
        "To cancel a subscription, please visit the subscriptions section of your account.",
        "If you need to cancel your service, please contact our customer support immediately.",
        "To cancel an order, please reach out to us before it's processed or shipped.",
        "Orders can be canceled before they are shipped. Once shipped, cancellations are not possible.",
        "If you would like to cancel your subscription, please do so before the next billing cycle.",
        "We allow cancellations within 7 days after purchase for a full refund.",
        "You can cancel your order via the 'Cancel' option on your order page.",
        "To cancel your account, please contact our support team for assistance.",
        "For cancellations, please provide your order ID and reason for cancellation."
    ],
    "feedback": [
        "We would love to hear your feedback! Please fill out our survey form.",
        "Your feedback is important to us. Please share your thoughts via email.",
        "How was your experience? We appreciate your feedback to improve our service.",
        "You can leave a review for our services on our website or through third-party platforms.",
        "Feel free to leave any feedback or suggestions through the contact form on our website.",
        "If you had a great experience, please consider sharing a positive review.",
        "We are always looking to improve. Tell us how we can make your experience better.",
        "To share your feedback, simply visit our 'Feedback' section on the website.",
        "We value your opinion! Please provide us with feedback so we can enhance our service.",
        "If you have any suggestions or comments, feel free to send them our way."
    ],
    "default": [
        "I'm not sure about that. Can you rephrase?",
        "I don't understand. Can you provide more details?",
        "Sorry, I couldn't get that. Could you clarify?",
        "I'm not sure how to answer that. Could you ask something else?",
        "That doesn't seem clear to me. Can you rephrase?",
        "Could you please provide more information or ask in another way?",
        "Sorry, I didn't quite get that. Can you try again?",
        "Can you clarify your request? I'm not quite sure what you mean.",
        "I didn't quite understand your question. Could you please explain again?",
        "Sorry, I couldn't understand that. Can you give me more details?"
    ]
}

# Predefined keywords for intent detection
keywords = {
    "booking": ["book", "schedule", "appointment", "reserve", "make an appointment", "book now", "reserve a spot", "schedule a meeting", "book a slot", "set an appointment"],
    "refund": ["refund", "money back", "return", "reimbursement", "get my money back", "refund process", "refund request", "return policy", "refund status", "order refund"],
    "support": ["help", "assist", "support", "customer service", "trouble", "problem", "issue", "support team", "technical support", "assistance"],
    "pricing": ["price", "cost", "charge", "fee", "pricing", "how much", "price details", "pricing plan", "cost breakdown", "subscription price"],
    "shipping": ["shipping", "delivery", "ship", "shipped", "shipping time", "delivery time", "shipping costs", "track order", "shipping status", "order tracking"],
    "account": ["account", "sign up", "login", "registration", "create account", "forgot password", "profile", "update account", "manage account", "sign in"],
    "payment": ["payment", "pay", "credit card", "purchase", "transaction", "checkout", "payment method", "billing", "payment gateway", "payment process"],
    "cancellation": ["cancel", "cancel order", "cancel subscription", "order cancellation", "service cancellation", "subscription cancel", "refund after cancellation", "cancel request", "cancel my order", "cancel my subscription"],
    "feedback": ["feedback", "review", "suggestion", "survey", "rate", "experience", "recommendation", "testimonial", "opinion", "comments"],
    "default": ["help", "info", "details", "question", "support", "assistance", "inquiry", "question about", "can you tell me", "what is"]
}

# Text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Track user's name
user_name = ""

def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes input text."""
    tokens = word_tokenize(text.lower())  # Tokenization & lowercase
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens


def detect_intent(tokens):
    """Detects user intent based on keywords in the query."""
    intent_scores = defaultdict(int)
    
    for token in tokens:
        for intent, words in keywords.items():
            if token in words:
                intent_scores[intent] += 1  # Increase score for matching intents
    
    if intent_scores:
        return max(intent_scores, key=intent_scores.get)  # Return the intent with the highest score
    return "default"  # If no match, return default response

def chatbot_response(user_input):
    """Processes user input and returns an appropriate chatbot response."""
    tokens = preprocess_text(user_input)  # Clean user input
    intent = detect_intent(tokens)  # Identify intent
    
    # Personalized greeting
    if user_name and "hello" in tokens or "hi" in tokens:
        return f"Hey {user_name}, how can I help you today?"

    # Default response
    return random.choice(responses[intent])  # Return a random response from the intent category

def get_user_name():
    """Asks for the user's name."""
    global user_name
    user_name = input("NPPD: Hi there! May I know your name? ")
    print(f"NPPD: Nice to meet you, {user_name}! How can I help you today?")
    

if __name__ == "__main__":
    print("NPPD Customer Support NPPD (type 'exit' to stop)")
    
    # Ask for the user's name at the start of the conversation
    get_user_name()
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print(f"NPPD: Thank you, {user_name}! Have a great day!")
            break
        reply = chatbot_response(user_input)
        print(f"NPPD: {reply}")