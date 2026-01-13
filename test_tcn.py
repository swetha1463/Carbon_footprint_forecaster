# Create test_tcn.py
import numpy as np
from models.tcn_module import TCNFeatureExtractor

# Create dummy data
X = np.random.randn(100, 50)  # 100 samples, 50 features
y = np.random.randn(100)

# Initialize and test
tcn = TCNFeatureExtractor(num_features=50, sequence_length=30)
X_tcn = tcn.fit_transform(X, y, epochs=10)

print(f"Input shape: {X.shape}")
print(f"Output shape: {X_tcn.shape}")
print("TCN module working! ✓")