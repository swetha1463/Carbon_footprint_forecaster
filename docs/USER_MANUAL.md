# User Manual

## Getting Started

### Dashboard Navigation

1. **Dashboard Page**: Overview of your carbon footprint
2. **Input Data**: Enter daily activities
3. **Forecast**: 7-day emission predictions
4. **Recommendations**: Personalized eco-actions
5. **Analytics**: Detailed statistics

### Entering Your Data

1. Navigate to "Input Data" page
2. Fill in daily activities:
   - Electricity usage (kWh)
   - Vehicle distance (km)
   - Waste generated (kg)
   - Water usage (liters)
3. Click "Calculate Footprint"
4. View results and recommendations

### Understanding Your GSI

**Green Score Index (GSI)** ranges from 0-100:
- **85-100**: Excellent (Top 15%)
- **70-84**: Good (Above average)
- **50-69**: Average
- **30-49**: Below average
- **0-29**: Needs improvement

### Taking Action

1. Review personalized recommendations
2. Mark actions as "Done" when completed
3. Track your progress over time
4. Watch your GSI improve!

## Troubleshooting

### Dashboard won't load
```bash
streamlit run app/dashboard.py --server.port 8502
```

### Model not found error
```bash
python main.py  # Re-run training
```

### Memory issues
Reduce `data.synthetic_samples` in `config.yaml`

## Support

Email: 2216163@saec.ac.in