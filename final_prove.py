import numpy as np
import json
from typing import List, Dict, Tuple
import time

class SentimentAnalysisAPI:
    
    def __init__(self, model_path='sentiment_model_improved.npz', vocab_path='vocabulary.json'):
        self.load_model(model_path)
        self.load_vocabulary(vocab_path)
        self.setup_response_templates()
        
    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        data = np.load(model_path)
        
        self.E = data['E']
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        self.vocab_size = int(data['vocab_size'])
        
        print(f"✓ Model loaded successfully (vocab_size: {self.vocab_size})")
        
    def load_vocabulary(self, vocab_path):
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            self.word_to_index = json.load(f)
        self.index_to_word = {int(idx): word for word, idx in self.word_to_index.items()}
        print(f"✓ Vocabulary loaded ({len(self.word_to_index)} words)")
        
    def setup_response_templates(self):
        self.response_templates = {
            'very_positive': {
                'tone': 'enthusiastic',
                'responses': [
                    "Wonderful! I'm thrilled to hear you're so happy! How can I help you today?",
                    "That's fantastic news! Your enthusiasm is contagious! What can I do for you?",
                    "Amazing! I love the positive energy! How may I assist you?"
                ],
                'emoji': '😊',
                'priority': 'celebrate'
            },
            'positive': {
                'tone': 'friendly',
                'responses': [
                    "Great to hear! How can I help you today?",
                    "That's nice! What can I do for you?",
                    "Glad you're having a good experience! How may I assist?"
                ],
                'emoji': '🙂',
                'priority': 'maintain'
            },
            'neutral': {
                'tone': 'professional',
                'responses': [
                    "I understand. How can I help you today?",
                    "Thank you for sharing. What can I assist you with?",
                    "I see. How may I help you?"
                ],
                'emoji': '👍',
                'priority': 'engage'
            },
            'negative': {
                'tone': 'empathetic',
                'responses': [
                    "I'm sorry to hear that. Let me help you resolve this issue.",
                    "I understand your frustration. What specifically can I help with?",
                    "That sounds challenging. Let's work together to fix this."
                ],
                'emoji': '😟',
                'priority': 'support'
            },
            'very_negative': {
                'tone': 'urgent_empathetic',
                'responses': [
                    "I sincerely apologize for this experience. Let me connect you with someone who can help immediately.",
                    "I'm very sorry you're going through this. Your satisfaction is our priority. Let me escalate this right away.",
                    "I completely understand your frustration. Let me get a specialist to help you immediately."
                ],
                'emoji': '🚨',
                'priority': 'escalate'
            }
        }
    
    def preprocess_text(self, text: str, max_length: int = 15) -> np.ndarray:
        text = text.lower().strip()
        words = text.split()
        
        indices = []
        for word in words[:max_length]:
            if word in self.word_to_index:
                indices.append(self.word_to_index[word])
            else:
                indices.append(self.word_to_index.get('<UNK>', 1))
        
        while len(indices) < max_length:
            indices.append(self.word_to_index.get('<PAD>', 0))
            
        return np.array([indices])
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    def forward_pass(self, X):
        embedded = self.E[X]
        mask = (X != 0).astype(float)
        mask_sum = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        pooled = np.sum(embedded * mask[:, :, np.newaxis], axis=1) / mask_sum
        z1 = np.dot(pooled, self.W1) + self.b1
        a1 = self.leaky_relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.leaky_relu(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        output = self.sigmoid(z3)
        
        return output[0, 0]
    
    def get_sentiment_category(self, probability: float) -> str:
        if probability >= 0.9:
            return 'very_positive'
        elif probability >= 0.6:
            return 'positive'
        elif probability >= 0.4:
            return 'neutral'
        elif probability >= 0.1:
            return 'negative'
        else:
            return 'very_negative'
    
    def analyze_sentiment(self, text: str) -> Dict:
        start_time = time.time()
        X = self.preprocess_text(text)
        probability = self.forward_pass(X)
        sentiment_label = 'positive' if probability > 0.5 else 'negative'
        confidence = abs(probability - 0.5) * 2
        category = self.get_sentiment_category(probability)
        template = self.response_templates[category]
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        
        return {
            'text': text,
            'sentiment': sentiment_label,
            'probability': float(probability),
            'confidence': float(confidence),
            'category': category,
            'response_template': template,
            'processing_time_ms': processing_time,
            'timestamp': time.time()
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        return [self.analyze_sentiment(text) for text in texts]
    
    def get_chatbot_response(self, user_message: str) -> Dict:
        analysis = self.analyze_sentiment(user_message)
        template = analysis['response_template']
        response_options = template['responses']
        selected_response = np.random.choice(response_options)
        
        return {
            'user_message': user_message,
            'sentiment_analysis': {
                'sentiment': analysis['sentiment'],
                'confidence': analysis['confidence'],
                'category': analysis['category']
            },
            'chatbot_response': selected_response,
            'tone': template['tone'],
            'priority': template['priority'],
            'should_escalate': template['priority'] == 'escalate',
            'processing_time_ms': analysis['processing_time_ms']
        }

# REST API Implementation (Flask example)
def create_flask_api(sentiment_api):
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy', 'model': 'loaded'})
    
    @app.route('/analyze', methods=['POST'])
    def analyze():
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        result = sentiment_api.analyze_sentiment(data['text'])
        return jsonify(result)
    
    @app.route('/batch_analyze', methods=['POST'])
    def batch_analyze():
        data = request.get_json()
        if 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400
        
        results = sentiment_api.batch_analyze(data['texts'])
        return jsonify({'results': results})
    
    @app.route('/chatbot_response', methods=['POST'])
    def chatbot_response():
        data = request.get_json()
        if 'message' not in data:
            return jsonify({'error': 'Missing message field'}), 400
        
        response = sentiment_api.get_chatbot_response(data['message'])
        return jsonify(response)
    
    return app

if __name__ == "__main__":
    api = SentimentAnalysisAPI()
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS API - READY FOR PRODUCTION")
    print("="*60)
    test_messages = [
        "This chatbot is amazing! Best customer service ever!",
        "I'm really happy with your quick response time",
        "Great product, exactly what I was looking for",
        "This is terrible, I want to speak to a manager",
        "Very disappointed with the service quality",
        "The product broke after one day of use",
        "I need help with my account settings",
        "The product is okay but shipping was slow",
        "Can you tell me about your return policy?"
    ]
    
    print("\nTEST RESULTS:")
    print("-"*60)
    
    for message in test_messages:
        response = api.get_chatbot_response(message)
        
        print(f"\nUser: '{message}'")
        print(f"Sentiment: {response['sentiment_analysis']['sentiment'].upper()} "
              f"({response['sentiment_analysis']['confidence']:.1%} confidence)")
        print(f"Category: {response['sentiment_analysis']['category']}")
        print(f"Chatbot: {response['chatbot_response']}")
        print(f"Tone: {response['tone']}")
        if response['should_escalate']:
            print("⚠️  ESCALATE TO HUMAN AGENT")
        print(f"Processing time: {response['processing_time_ms']:.1f}ms")
    
    print("\n" + "="*60)
    print("API ENDPOINTS:")
    print("POST /analyze - Analyze single text")
    print("POST /batch_analyze - Analyze multiple texts")
    print("POST /chatbot_response - Get chatbot response with sentiment")
    print("GET /health - Health check")
    
    # Performance metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS:")
    import time
    times = []
    for _ in range(100):
        start = time.time()
        api.analyze_sentiment("This is a test message for performance")
        times.append((time.time() - start) * 1000)
    
    print(f"Average processing time: {np.mean(times):.1f}ms")
    print(f"95th percentile: {np.percentile(times, 95):.1f}ms")
    print(f"Max processing time: {np.max(times):.1f}ms")
    config = {
        'model_path': 'sentiment_model_improved.npz',
        'vocab_path': 'vocabulary.json',
        'max_text_length': 15,
        'confidence_threshold': 0.8,
        'escalation_threshold': 0.1,
        'supported_languages': ['en'],
        'api_version': '1.0.0'
    }
    
    with open('api_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ API configuration saved to 'api_config.json'")
    print("✅ Ready for deployment!")