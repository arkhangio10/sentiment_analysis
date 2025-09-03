# Sentiment Analysis Neural Network from Scratch

A production-ready sentiment analysis system built entirely from scratch using only NumPy. This project demonstrates deep understanding of neural network fundamentals by implementing forward propagation, backpropagation, and gradient descent without any ML frameworks.

## 🌟 Features

- **Pure NumPy Implementation** - No TensorFlow, PyTorch, or scikit-learn
- **Custom Neural Network** - 3-layer architecture with embedding layer
- **Fast Inference** - <10ms per prediction
- **Production API** - REST endpoints for easy integration
- **Automatic Escalation** - Detects highly negative sentiments for human intervention
- **Customizable Dataset** - Easy to adapt for any domain

## 📊 Performance

- **Validation Accuracy**: 81.18%
- **F1-Score**: 80.00%
- **Precision**: 82.05%
- **Recall**: 78.05%
- **Processing Time**: <10ms per prediction

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-from-scratch.git
cd sentiment-analysis-from-scratch

Install dependencies:

bashpip install numpy matplotlib flask

Train the model:

bashpython analysis_sentiment.py  # Create dataset
python training.py            # Train neural network

Run the API:

bashpython step4_api.py
📁 Project Structure
sentiment-analysis-from-scratch/
├── analysis_sentiment.py    # Dataset creation and vocabulary building
├── training.py             # Neural network training
├── step4_api.py           # Production API
├── network.py             # Basic network architecture
├── vocabulary.json        # Word-to-index mappings
├── sentiment_model_improved.npz  # Trained model weights
└── README.md
🛠️ Technical Implementation
Neural Network Architecture

Embedding Layer: 100-dimensional word embeddings
Hidden Layer 1: 256 neurons with Leaky ReLU activation
Hidden Layer 2: 128 neurons with Leaky ReLU activation
Output Layer: 1 neuron with Sigmoid activation

Key Features

Dropout Regularization: Prevents overfitting
L2 Regularization: Improves generalization
Momentum Optimizer: Faster convergence
Data Augmentation: Random word masking
Early Stopping: Prevents overtraining

📖 API Usage
Analyze Single Text
bashcurl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
Batch Analysis
bashcurl -X POST http://localhost:5000/batch_analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great service!", "Terrible experience"]}'
Chatbot Response
bashcurl -X POST http://localhost:5000/chatbot_response \
  -H "Content-Type: application/json" \
  -d '{"message": "I am very frustrated with this product"}'
🎯 Customizing for Your Domain

Open analysis_sentiment.py
Replace the positive_texts and negative_texts arrays:

pythonpositive_texts = [
    "your positive example 1",
    "your positive example 2",
    # ... add more
]

negative_texts = [
    "your negative example 1", 
    "your negative example 2",
    # ... add more
]

Rebuild vocabulary and retrain:

bashpython analysis_sentiment.py
python training.py
📈 Applications

Customer Service Chatbots - Detect frustrated customers
Review Analysis - Categorize product reviews
Social Media Monitoring - Track brand sentiment
Support Ticket Routing - Prioritize urgent issues
Employee Feedback - Monitor workplace satisfaction

🔧 Understanding the Code
1. Data Preprocessing (analysis_sentiment.py)

Builds vocabulary from training texts
Converts words to numerical indices
Creates balanced dataset (50% positive, 50% negative)

2. Neural Network Training (training.py)

Implements forward propagation
Calculates gradients with backpropagation
Updates weights using momentum optimizer
Validates on held-out data

3. Production API (step4_api.py)

Loads trained model
Preprocesses input text
Returns sentiment with confidence score
Suggests appropriate chatbot responses

🔮 Future Improvements

 Attention mechanism for context understanding
 Multi-language support
 Character-level embeddings for typo handling
 Pre-trained word embeddings (GloVe/Word2Vec)
 Real-time learning from feedback
 Fine-grained sentiment (1-5 scale)
 Aspect-based sentiment analysis

🤝 Contributing
Contributions are welcome! Feel free to:

Add more training data
Improve model architecture
Add new features
Fix bugs
Improve documentation

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Built as a learning project to understand neural networks from scratch
Inspired by the need for lightweight, customizable sentiment analysis
Special thanks to the NumPy team for their excellent library

📧 Contact

LinkedIn: https://www.linkedin.com/in/abel-brayan-mancilla-montesinos-9b2a88132/
GitHub: arkhangio10



Made with ❤️ by ABEL Mancilla
