"""Temporal Convolutional Network for time series patterns."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np

class Chomp1d(nn.Module):
    """Remove extra padding from causal convolution."""
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Basic building block of TCN."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, 
                 padding, dropout=0.2):
        """
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Convolution kernel size
            stride: Stride for convolution
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout probability
        """
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, 
                                          dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, 
                                          dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor
        
        Returns:
            Output after residual connection
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            num_inputs: Number of input features
            num_channels: List of channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                    stride=1, dilation=dilation_size,
                                    padding=(kernel_size-1) * dilation_size, 
                                    dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, features, sequence_length)
        
        Returns:
            Output tensor
        """
        return self.network(x)


class TCNFeatureExtractor:
    """TCN-based feature extractor for time series."""
    
    def __init__(self, num_features: int, num_channels: list = None,
                 kernel_size: int = 3, dropout: float = 0.2, 
                 sequence_length: int = 30, device: str = None):
        """
        Args:
            num_features: Number of input features
            num_channels: List of channel sizes
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            sequence_length: Length of input sequences
            device: Torch device
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.sequence_length = sequence_length
        
        if num_channels is None:
            num_channels = [64, 128, 256]
        
        self.model = TemporalConvNet(num_features, num_channels, 
                                    kernel_size, dropout).to(self.device)
        
        # Output dimension
        self.output_dim = num_channels[-1]
        
        # Projection layer to match original feature dimension
        self.projection = nn.Linear(self.output_dim, num_features).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projection.parameters()),
            lr=0.001
        )
        
        self.is_fitted = False
        
    def create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences from feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Sequences (n_sequences, n_features, sequence_length)
        """
        n_samples, n_features = X.shape
        sequences = []
        
        # Handle case where n_samples < sequence_length
        if n_samples < self.sequence_length:
            # Pad with zeros at the beginning
            padding = np.zeros((self.sequence_length - n_samples, n_features))
            X_padded = np.vstack([padding, X])
            seq = X_padded.T  # (features, time)
            sequences.append(seq)
        else:
            for i in range(n_samples - self.sequence_length + 1):
                seq = X[i:i + self.sequence_length, :].T  # (features, time)
                sequences.append(seq)
        
        return np.array(sequences)
    
    def fit(self, X: np.ndarray, y: np.ndarray = None, epochs: int = 50):
        """Train TCN feature extractor.
        
        Args:
            X: Training features
            y: Target values
            epochs: Number of training epochs
        """
        self.model.train()
        
        # Create sequences
        X_seq = self.create_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        if y is not None:
            # Handle y padding for small datasets
            if len(y) < self.sequence_length:
                y_seq = np.array([y[-1]])  # Use last value
            else:
                y_seq = y[self.sequence_length - 1:]
            y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            tcn_output = self.model(X_tensor)
            
            # Take last time step
            features = tcn_output[:, :, -1]
            
            # Project back to original dimension
            projected = self.projection(features)
            
            # Reconstruction loss
            original = X_tensor[:, :, -1].transpose(0, 1)
            reconstruction_loss = nn.MSELoss()(projected.T, original)
            
            # Supervised loss if labels provided
            if y is not None:
                prediction = torch.mean(projected, dim=1)
                supervised_loss = nn.MSELoss()(prediction, y_tensor)
                loss = reconstruction_loss + 0.5 * supervised_loss
            else:
                loss = reconstruction_loss
            
            loss.backward()
            self.optimizer.step()
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract TCN features.
        
        Args:
            X: Input features
        
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise RuntimeError("TCNFeatureExtractor must be fitted before transform")
            
        self.model.eval()
        
        with torch.no_grad():
            n_samples = X.shape[0]
            X_seq = self.create_sequences(X)
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            # Forward pass
            tcn_output = self.model(X_tensor)
            
            # Take last time step and project
            features = tcn_output[:, :, -1]
            projected = self.projection(features)
            
            result = projected.cpu().numpy()
            
            # Pad to match original length
            if n_samples >= self.sequence_length:
                padding = np.zeros((self.sequence_length - 1, self.num_features))
                result = np.vstack([padding, result])
            else:
                # If we padded the input, repeat the single output for all samples
                result = np.repeat(result, n_samples, axis=0)
            
            return result
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, 
                     epochs: int = 50) -> np.ndarray:
        """Fit and transform features.
        
        Args:
            X: Training features
            y: Target values
            epochs: Number of epochs
        
        Returns:
            Transformed features
        """
        self.fit(X, y, epochs)
        return self.transform(X)