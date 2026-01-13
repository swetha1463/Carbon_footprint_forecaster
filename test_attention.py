# Create test_attention.py
import numpy as np
from models.attention_module import AttentionWeighter

# Create dummy data
X = np.random.randn(100, 50)  # 100 samples, 50 features
y = np.random.randn(100)

# Initialize and test
attention = AttentionWeighter(input_dim=50)
X_weighted = attention.fit_transform(X, y, epochs=10)

print(f"Input shape: {X.shape}")
print(f"Output shape: {X_weighted.shape}")
print("Attention module working! ✓")