import nltk
from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define chatbot pairs (patterns and responses)
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, nice to meet you!",
         "Hi %1, pleased to make your acquaintance.",
         "Nice to meet you, %1!"]
    ],
    [
        r"what is your name?",
        ["I'm a chatbot created to assist you.",
         "You can call me ChatBot.",
         "I'm ChatBot, your virtual assistant."]
    ],
    [
        r"how are you?",
        ["I'm doing well, thank you for asking!",
         "I'm great! How can I help you today?",
         "Excellent! Ready to assist you."]
    ],
    [
        r"what can you do?",
        ["I can help you with information, answer questions, and have conversations.",
         "I'm here to assist with various topics and provide helpful information.",
         "I can chat, answer questions, and provide assistance on many topics."]
    ],
    [
        r"tell me about (.*)",
        ["I'd be happy to help! %1 is an interesting topic.",
         "Let me help you with information about %1.",
         "%1 is a fascinating subject. What specifically would you like to know?"]
    ],
    [
        r"what is (.*)?",
        ["%1 is an interesting concept.",
         "Great question about %1!",
         "I can help explain %1 to you."]
    ],
    [
        r"(.*) weather",
        ["I don't have real-time weather data, but you can check a weather service.",
         "For weather information, please check your local weather service.",
         "You can find weather updates on weather.com or similar services."]
    ],
    [
        r"(.*) (location|city)?",
        ["I'm a virtual chatbot, so I exist everywhere and nowhere!",
         "I'm here to help wherever you are!",
         "Location isn't important for our conversation."]
    ],
    [
        r"(.*) (help|assist)",
        ["I'm here to help! What do you need assistance with?",
         "Of course! How can I assist you?",
         "I'm ready to help. What's your question?"]
    ],
    [
        r"(.*) (hello|hi|hey|greetings)",
        ["Hello! How can I help you today?",
         "Hi there! What can I do for you?",
         "Hey! Great to see you here."]
    ],
    [
        r"(.*) (goodbye|bye|see you|farewell)",
        ["Goodbye! Have a great day!",
         "Bye! Thanks for chatting with me.",
         "See you later! Take care!"]
    ],
    [
        r"(.*)",
        ["That's interesting! Tell me more.",
         "I see. Can you provide more details?",
         "Interesting point. What else would you like to discuss?",
         "Got it! Anything else I can help with?"]
    ]
]

# Reflections for pronoun replacement
reflections.update({
    "i am": "you are",
    "i'm": "you're",
    "i have": "you have",
    "i've": "you've",
    "i will": "you will",
    "i'll": "you'll",
    "my": "your",
    "me": "you",
    "i": "you"
})

class CustomChatBot:
    def __init__(self, pairs, reflections):
        self.chat = Chat(pairs, reflections)
        self.conversation_history = []

    def chat_with_user(self):
        """
        Main chatbot conversation loop
        """
        print("\n" + "="*60)
        print("Welcome to ChatBot!")
        print("="*60)
        print("Type 'quit' to exit the conversation\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("ChatBot: Goodbye! Thanks for chatting with me!\n")
                    break

                # Get response from chatbot
                response = self.chat.respond(user_input)

                # Store conversation history
                self.conversation_history.append({
                    'user': user_input,
                    'bot': response
                })

                print(f"ChatBot: {response}\n")

            except KeyboardInterrupt:
                print("\nChatBot: Goodbye!\n")
                break
            except Exception as e:
                print(f"Error: {e}")

    def get_conversation_history(self):
        """
        Return conversation history
        """
        return self.conversation_history

    def save_conversation(self, filename='conversation_log.txt'):
        """
        Save conversation to a text file
        """
        with open(filename, 'w') as f:
            for i, exchange in enumerate(self.conversation_history, 1):
                f.write(f"Turn {i}:\n")
                f.write(f"User: {exchange['user']}\n")
                f.write(f"ChatBot: {exchange['bot']}\n")
                f.write("-" * 40 + "\n")

        print(f"✓ Conversation saved to {filename}")

# Main execution
if __name__ == "__main__":
    chatbot = CustomChatBot(pairs, reflections)
    chatbot.chat_with_user()

    # Save conversation log
    if chatbot.get_conversation_history():
        chatbot.save_conversation()
