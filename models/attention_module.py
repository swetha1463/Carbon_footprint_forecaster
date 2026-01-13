"""Attention mechanism for feature weighting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature importance."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.output = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Weighted features and attention weights
        """
        batch_size = x.shape[0]
        
        # Linear projections
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.view(batch_size, self.hidden_dim)
        
        # Output projection
        output = self.output(context)
        
        return output, attention_weights.mean(dim=1)


class FeatureAttention(nn.Module):
    """Attention-based feature weighting for XGBoost."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
        """
        super(FeatureAttention, self).__init__()
        
        self.attention = MultiHeadAttention(input_dim, hidden_dim, num_heads)
        
        # Feature importance network
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, num_features)
        
        Returns:
            Weighted features and importance scores
        """
        # Apply attention
        attended_features, attention_weights = self.attention(x)
        
        # Calculate feature importance
        importance = self.importance_net(x)
        
        # Weight features by importance
        weighted_features = x * importance + attended_features * (1 - importance)
        
        return weighted_features, importance


class AttentionWeighter:
    """Wrapper for attention-based feature weighting compatible with sklearn."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_heads: int = 4, device: str = None):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            device: Torch device
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FeatureAttention(input_dim, hidden_dim, num_heads).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray = None, epochs: int = 50):
        """Train attention weights.
        
        Args:
            X: Training features
            y: Target values (optional, used for supervised weighting)
            epochs: Number of training epochs
        """
        self.model.train()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            weighted_features, importance = self.model(X_tensor)
            
            # Reconstruction loss
            reconstruction_loss = F.mse_loss(weighted_features, X_tensor)
            
            # Sparsity penalty for importance
            sparsity_loss = torch.mean(importance)
            
            # Supervised loss if labels provided
            if y is not None:
                # Simple linear prediction for supervision
                prediction = torch.sum(weighted_features, dim=1)
                supervised_loss = F.mse_loss(prediction, y_tensor)
                loss = reconstruction_loss + 0.1 * sparsity_loss + 0.5 * supervised_loss
            else:
                loss = reconstruction_loss + 0.1 * sparsity_loss
            
            loss.backward()
            self.optimizer.step()
        
        self.trained = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply attention weighting to features.
        
        Args:
            X: Input features
        
        Returns:
            Weighted features
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            weighted_features, _ = self.model(X_tensor)
            return weighted_features.cpu().numpy()
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, 
                     epochs: int = 50) -> np.ndarray:
        """Fit and transform features.
        
        Args:
            X: Training features
            y: Target values
            epochs: Number of epochs
        
        Returns:
            Weighted features
        """
        self.fit(X, y, epochs)
        return self.transform(X)
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get feature importance scores.
        
        Args:
            X: Input features
        
        Returns:
            Importance scores for each feature
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, importance = self.model(X_tensor)
            return importance.cpu().numpy().mean(axis=0)