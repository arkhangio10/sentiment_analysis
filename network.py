import numpy as np

data = np.load('step1_data.npz')
X = data['X']
y = data['y']
vocab_size = data['vocab_size']

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

class NeuralNetwork:
    def __init__(self, vocab_size, embedding_dim=10, hidden_dim=16, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        self.E = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.W1 = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(2.0 / embedding_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))
    
    def forward(self, X):
        batch_size, seq_length = X.shape
        
        embedded = np.zeros((batch_size, seq_length, self.embedding_dim))
        for i in range(batch_size):
            for j in range(seq_length):
                if X[i, j] != 0:
                    embedded[i, j] = self.E[X[i, j]]
        
        mask = (X != 0).astype(float)
        mask_sum = mask.sum(axis=1, keepdims=True) + 1e-8
        
        pooled = np.zeros((batch_size, self.embedding_dim))
        for i in range(batch_size):
            for j in range(seq_length):
                if mask[i, j]:
                    pooled[i] += embedded[i, j]
            pooled[i] /= mask_sum[i]
        
        z1 = np.dot(pooled, self.W1) + self.b1
        a1 = relu(z1)
        
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        
        cache = {
            'X': X, 'embedded': embedded, 'pooled': pooled,
            'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2,
            'mask': mask, 'mask_sum': mask_sum
        }
        
        return a2, cache
    
    def compute_loss(self, y_pred, y_true):
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, cache, y_true):
        batch_size = y_true.shape[0]
        
        X = cache['X']
        pooled = cache['pooled']
        z1 = cache['z1']
        a1 = cache['a1']
        a2 = cache['a2']
        mask_sum = cache['mask_sum']
        
        dz2 = a2 - y_true.reshape(-1, 1)
        
        dW2 = np.dot(a1.T, dz2) / batch_size
        db2 = np.mean(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_derivative(z1)
        
        dW1 = np.dot(pooled.T, dz1) / batch_size
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        dpooled = np.dot(dz1, self.W1.T)
        
        dE = np.zeros_like(self.E)
        for i in range(batch_size):
            for j in range(X.shape[1]):
                if X[i, j] != 0:
                    dE[X[i, j]] += dpooled[i] / mask_sum[i]
        
        dE /= batch_size
        
        gradients = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dE': dE
        }
        
        return gradients
    
    def update_weights(self, gradients):
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
        self.E -= self.learning_rate * gradients['dE']
    
    def train_step(self, X_batch, y_batch):
        predictions, cache = self.forward(X_batch)
        loss = self.compute_loss(predictions, y_batch)
        gradients = self.backward(cache, y_batch)
        self.update_weights(gradients)
        return loss, predictions
    
    def predict(self, X):
        predictions, _ = self.forward(X)
        return predictions

print("STEP 2: NEURAL NETWORK ARCHITECTURE")
print("=" * 50)
print(f"Input: {X.shape[1]} words per text")
print(f"Embedding: {vocab_size} words → 10D vectors")
print(f"Hidden Layer: 10D → 16 neurons (ReLU)")
print(f"Output Layer: 16 → 1 neuron (Sigmoid)")
print("\nTotal parameters:")

model = NeuralNetwork(vocab_size=int(vocab_size), embedding_dim=10, hidden_dim=16)

total_params = (
    model.E.shape[0] * model.E.shape[1] +
    model.W1.shape[0] * model.W1.shape[1] + model.b1.shape[1] +
    model.W2.shape[0] * model.W2.shape[1] + model.b2.shape[1]
)
print(f"  {total_params} trainable parameters")

print("\nINITIAL PREDICTIONS (before training):")
initial_pred, _ = model.forward(X)
print(f"Min: {initial_pred.min():.4f}, Max: {initial_pred.max():.4f}")

np.savez('step2_model_init.npz',
         E=model.E, W1=model.W1, b1=model.b1, 
         W2=model.W2, b2=model.b2)

print("\nModel initialized and saved to 'step2_model_init.npz'")
