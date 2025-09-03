import numpy as np
import matplotlib.pyplot as plt

data = np.load('sentiment_data_fixed.npz')
X = data['X']
y = data['y']
vocab_size = data['vocab_size']

print(f"Data loaded successfully!")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Vocabulary size: {vocab_size}")

print(f"\nClass distribution:")
print(f"Positive samples: {np.sum(y == 1)} ({np.mean(y == 1):.1%})")
print(f"Negative samples: {np.sum(y == 0)} ({np.mean(y == 0):.1%})")

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def dropout(x, rate=0.3, training=True):
    if not training:
        return x, None
    mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
    return x * mask, mask
class ImprovedNeuralNetwork:
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, hidden_dim2=128, 
                 learning_rate=0.05, l2_lambda=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        
        self.E = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        self.W1 = np.random.normal(0, np.sqrt(2.0/embedding_dim), (embedding_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.normal(0, np.sqrt(2.0/hidden_dim), (hidden_dim, hidden_dim2))
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = np.random.normal(0, np.sqrt(1.0/hidden_dim2), (hidden_dim2, 1))
        self.b3 = np.zeros((1, 1))
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_W3 = np.zeros_like(self.W3)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_b2 = np.zeros_like(self.b2)
        self.v_b3 = np.zeros_like(self.b3)
        self.v_E = np.zeros_like(self.E)
        self.momentum = 0.9
    
    def forward(self, X, training=True):
        batch_size, seq_length = X.shape
        embedded = self.E[X]
        embedded_dropout, _ = dropout(embedded, rate=0.1, training=training)
        mask = (X != 0).astype(float)
        mask_sum = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        pooled = np.sum(embedded_dropout * mask[:, :, np.newaxis], axis=1) / mask_sum
        z1 = np.dot(pooled, self.W1) + self.b1
        a1 = leaky_relu(z1)
        a1_dropout, a1_mask = dropout(a1, rate=0.3, training=training)
        z2 = np.dot(a1_dropout, self.W2) + self.b2
        a2 = leaky_relu(z2)
        a2_dropout, a2_mask = dropout(a2, rate=0.3, training=training)
        z3 = np.dot(a2_dropout, self.W3) + self.b3
        a3 = sigmoid(z3)
        
        cache = {
            'X': X, 'pooled': pooled, 'embedded': embedded,
            'z1': z1, 'a1': a1, 'a1_dropout': a1_dropout, 'a1_mask': a1_mask,
            'z2': z2, 'a2': a2, 'a2_dropout': a2_dropout, 'a2_mask': a2_mask,
            'z3': z3, 'a3': a3, 'mask': mask, 'mask_sum': mask_sum
        }
        
        return a3, cache
    
    def compute_loss(self, y_pred, y_true):
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        ce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2_loss = 0.5 * self.l2_lambda * (
            np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2)
        )
        
        return ce_loss + l2_loss
    
    def backward(self, cache, y_true):
        batch_size = y_true.shape[0]
        X = cache['X']
        pooled = cache['pooled']
        z1, a1_mask = cache['z1'], cache['a1_mask']
        z2, a2_mask = cache['z2'], cache['a2_mask']
        a1_dropout = cache['a1_dropout']
        a2_dropout = cache['a2_dropout']
        a3 = cache['a3']
        mask = cache['mask']
        mask_sum = cache['mask_sum']
        
        
        dz3 = (a3 - y_true.reshape(-1, 1)) / batch_size
        dW3 = np.dot(a2_dropout.T, dz3) + self.l2_lambda * self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        
        da2_dropout = np.dot(dz3, self.W3.T)
        da2 = da2_dropout * a2_mask if a2_mask is not None else da2_dropout
        dz2 = da2 * leaky_relu_derivative(z2)
        dW2 = np.dot(a1_dropout.T, dz2) + self.l2_lambda * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
       
        da1_dropout = np.dot(dz2, self.W2.T)
        da1 = da1_dropout * a1_mask if a1_mask is not None else da1_dropout
        dz1 = da1 * leaky_relu_derivative(z1)
        dW1 = np.dot(pooled.T, dz1) + self.l2_lambda * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        
        dpooled = np.dot(dz1, self.W1.T)
        dE = np.zeros_like(self.E)
        
        for i in range(batch_size):
            for j in range(X.shape[1]):
                if X[i, j] != 0:
                    dE[X[i, j]] += dpooled[i] / mask_sum[i]
        
        gradients = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3,
            'dE': dE
        }
        
        return gradients
    
    def update_weights(self, gradients):
        
        self.v_W1 = self.momentum * self.v_W1 - self.learning_rate * gradients['dW1']
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * gradients['dW2']
        self.v_W3 = self.momentum * self.v_W3 - self.learning_rate * gradients['dW3']
        self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * gradients['db1']
        self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * gradients['db2']
        self.v_b3 = self.momentum * self.v_b3 - self.learning_rate * gradients['db3']
        self.v_E = self.momentum * self.v_E - self.learning_rate * gradients['dE']
        
        self.W1 += self.v_W1
        self.W2 += self.v_W2
        self.W3 += self.v_W3
        self.b1 += self.v_b1
        self.b2 += self.v_b2
        self.b3 += self.v_b3
        self.E += self.v_E
    
    def train_step(self, X_batch, y_batch):
        predictions, cache = self.forward(X_batch, training=True)
        loss = self.compute_loss(predictions, y_batch)
        gradients = self.backward(cache, y_batch)
        self.update_weights(gradients)
        return loss, predictions
    
    def predict(self, X):
        predictions, _ = self.forward(X, training=False)
        return predictions


def augment_text_indices(X, prob=0.1):
    """Randomly mask some words to create augmented samples"""
    X_aug = X.copy()
    mask = np.random.random(X.shape) < prob
    mask = mask & (X != 0)  
    X_aug[mask] = 1  
    return X_aug


np.random.seed(42)
positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]


train_pos = np.random.choice(positive_indices, size=int(0.8 * len(positive_indices)), replace=False)
train_neg = np.random.choice(negative_indices, size=int(0.8 * len(negative_indices)), replace=False)
train_indices = np.concatenate([train_pos, train_neg])


val_indices = np.setdiff1d(np.arange(len(X)), train_indices)

X_train = X[train_indices]
y_train = y[train_indices]
X_val = X[val_indices]
y_val = y[val_indices]

print(f"\nTraining samples: {len(X_train)} (Pos: {np.sum(y_train == 1)}, Neg: {np.sum(y_train == 0)})")
print(f"Validation samples: {len(X_val)} (Pos: {np.sum(y_val == 1)}, Neg: {np.sum(y_val == 0)})")


model = ImprovedNeuralNetwork(
    vocab_size=int(vocab_size),
    embedding_dim=100,
    hidden_dim=256,
    hidden_dim2=128,
    learning_rate=0.05,
    l2_lambda=0.001
)

print("\nTRAINING IMPROVED NEURAL NETWORK")
print("=" * 60)


epochs = 1000
batch_size = 64
best_val_acc = 0
patience = 100
patience_counter = 0
best_model_weights = None


initial_lr = 0.05
lr_decay = 0.95
min_lr = 0.0001


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


for epoch in range(epochs):

    shuffle_idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffle_idx]
    y_train_shuffled = y_train[shuffle_idx]
    
    
    epoch_losses = []
    for start in range(0, len(X_train), batch_size):
        end = min(start + batch_size, len(X_train))
        
       
        if np.random.random() < 0.5:
            X_batch = augment_text_indices(X_train_shuffled[start:end])
        else:
            X_batch = X_train_shuffled[start:end]
        
        y_batch = y_train_shuffled[start:end]
        
        loss, _ = model.train_step(X_batch, y_batch)
        epoch_losses.append(loss)
    

    train_loss = np.mean(epoch_losses)
    train_pred = model.predict(X_train)
    train_acc = np.mean((train_pred > 0.5).flatten() == y_train)
    

    val_pred = model.predict(X_val)
    val_loss = model.compute_loss(val_pred, y_val)
    val_acc = np.mean((val_pred > 0.5).flatten() == y_val)
    

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        best_model_weights = {
            'E': model.E.copy(),
            'W1': model.W1.copy(), 'b1': model.b1.copy(),
            'W2': model.W2.copy(), 'b2': model.b2.copy(),
            'W3': model.W3.copy(), 'b3': model.b3.copy()
        }
    else:
        patience_counter += 1
        

        if patience_counter % 20 == 0 and model.learning_rate > min_lr:
            model.learning_rate *= lr_decay
            model.learning_rate = max(model.learning_rate, min_lr)
    

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | LR: {model.learning_rate:.5f}")
    
  
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break


if best_model_weights:
    model.E = best_model_weights['E']
    model.W1 = best_model_weights['W1']
    model.b1 = best_model_weights['b1']
    model.W2 = best_model_weights['W2']
    model.b2 = best_model_weights['b2']
    model.W3 = best_model_weights['W3']
    model.b3 = best_model_weights['b3']


print("\nFINAL RESULTS:")
print(f"Best Validation Accuracy: {best_val_acc:.2%}")
print(f"Final Train Accuracy: {train_accuracies[-1]:.2%}")


val_pred_final = model.predict(X_val)
val_pred_labels = (val_pred_final > 0.5).flatten()


from collections import Counter
tp = np.sum((val_pred_labels == 1) & (y_val == 1))
tn = np.sum((val_pred_labels == 0) & (y_val == 0))
fp = np.sum((val_pred_labels == 1) & (y_val == 0))
fn = np.sum((val_pred_labels == 0) & (y_val == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nValidation Metrics:")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
print(f"\nConfusion Matrix:")
print(f"True Pos: {tp}, True Neg: {tn}")
print(f"False Pos: {fp}, False Neg: {fn}")


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.7)
plt.plot(val_losses, label='Val Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Acc', alpha=0.7)
plt.plot(val_accuracies, label='Val Acc', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.4, 1.05)

plt.subplot(1, 3, 3)

plt.hist(val_pred_final[y_val == 1], bins=20, alpha=0.5, label='Positive', density=True)
plt.hist(val_pred_final[y_val == 0], bins=20, alpha=0.5, label='Negative', density=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Prediction Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improved_training_results.png', dpi=150)
print("\nPlots saved to 'improved_training_results.png'")


np.savez('sentiment_model_improved.npz',
         E=model.E, W1=model.W1, b1=model.b1,
         W2=model.W2, b2=model.b2, W3=model.W3, b3=model.b3,
         vocab_size=vocab_size)

print("\nImproved model saved to 'sentiment_model_improved.npz'")


print("\nSAMPLE PREDICTIONS:")
print("-" * 60)
for i in range(min(15, len(X_val))):
    pred = val_pred_final[i, 0]
    actual = y_val[i]
    pred_label = "POSITIVE" if pred > 0.5 else "NEGATIVE"
    actual_label = "POSITIVE" if actual == 1 else "NEGATIVE"
    confidence = abs(pred - 0.5) * 2  # Convert to 0-1 confidence
    correct = "✓" if (pred > 0.5) == actual else "✗"
    print(f"{correct} Pred: {pred_label} ({pred:.3f}, conf: {confidence:.1%}) | Actual: {actual_label}")