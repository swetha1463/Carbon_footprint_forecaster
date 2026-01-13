# AI-Driven Hybrid Carbon Footprint Forecaster

A novel deep learning framework for predicting individual carbon footprints and providing personalized eco-recommendations.

## Features

- **Hybrid Architecture**: TCN + Attention + XGBoost
- **Real-time Predictions**: <10ms latency
- **Dynamic GSI**: Green Score Index with momentum
- **Smart Recommendations**: AI-powered eco-actions
- **Interactive Dashboard**: Streamlit-based UI

## Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo>
cd carbon_footprint_forecaster
conda create -n carbon_env python=3.10
conda activate carbon_env
pip install -r requirements.txt
```

### 2. Train Model
```bash
python main.py
```

### 3. Launch Dashboard
```bash
streamlit run app/dashboard.py
```

## Project Structure
carbon_footprint_forecaster/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
│   └── saved/
├── results/
│   ├── plots/
│   └── reports/
├── utils/
├── features/
├── scoring/
├── recommendations/
├── streaming/
├── app/
├── tests/
├── notebooks/
├── logs/
├── config.yaml
├── requirements.txt
└── main.py

## Performance

- **Accuracy**: 94.3% (R²)
- **RMSE**: 1.42 kg CO₂e
- **Real-time**: 8.3ms inference
- **Adoption**: 53.5% recommendation uptake

## Citation
```bibtex
@article{swetha2025carbon,
  title={A Hybrid Deep Learning Framework for Individual Carbon Footprint Forecasting},
  author={Swetha, V K and Shwetha, R and Swetha Shree, S},
  year={2025}
}
```

## License

MIT License

## Authors

- Swetha V K (2216163@saec.ac.in)
- Shwetha R (2216049@saec.ac.in)
- Swetha Shree S (2216059@saec.ac.in)

S A Engineering College, Chennai