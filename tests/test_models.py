"""Test suite for models."""
import numpy as np
import pytest
from models.hybrid_model import HybridCarbonForecaster

def test_hybrid_model_initialization():
    """Test model can be initialized."""
    model = HybridCarbonForecaster(
        num_features=50,
        use_tcn=True,
        use_attention=True
    )
    assert model is not None
    assert model.num_features == 50

def test_hybrid_model_training():
    """Test model can be trained."""
    # Create dummy data
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    
    model = HybridCarbonForecaster(num_features=50)
    model.fit(X, y)
    
    # Check predictions
    y_pred = model.predict(X[:10])
    assert len(y_pred) == 10
    assert not np.isnan(y_pred).any()

def test_hybrid_model_save_load():
    """Test model can be saved and loaded."""
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    
    # Train and save
    model = HybridCarbonForecaster(num_features=50)
    model.fit(X, y)
    model.save('tests/temp_model.pkl')
    
    # Load and test
    loaded_model = HybridCarbonForecaster.load('tests/temp_model.pkl')
    y_pred_original = model.predict(X[:10])
    y_pred_loaded = loaded_model.predict(X[:10])
    
    assert np.allclose(y_pred_original, y_pred_loaded)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])