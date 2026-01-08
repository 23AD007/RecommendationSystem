# Packaging Recommendation System

An AI-powered system for recommending sustainable packaging materials based on product specifications and environmental impact analysis.

## Features

- **AI-Powered Recommendations**: Machine learning models for intelligent packaging material suggestions
- **Dynamic Web Interface**: Interactive Streamlit frontend for easy product specification input
- **REST API**: Flask-based API for programmatic access
- **Environmental Impact Analysis**: Sustainability scoring and eco-friendly material recommendations
- **MLflow Integration**: Experiment tracking and model management
- **Database Integration**: PostgreSQL support for data persistence

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HTML Frontend │    │     Flask       │    │   PostgreSQL    │
│   (Port 8000)   │◄──►│     API         │◄──►│   Database      │
│                 │    │   (Port 5000)  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                          │
                          ▼
                   ┌─────────────────┐
                   │   ML Models     │
                   │  (MLflow)       │
                   └─────────────────┘
```

Alternative: Streamlit Frontend (Port 8501)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Packaging-Recommendation-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/ecopackdb
   API_KEY=your-secret-api-key
   MLFLOW_TRACKING_URI=http://localhost:5000
   ```

## Usage

### Running the Backend API

1. **Start the Flask API server**
   ```bash
   python -m src.api.app
   ```
   The API will be available at `http://localhost:5000`

2. **API Endpoints**
   - `GET /health` - Health check
   - `POST /api/product/recommend-materials` - Get material recommendations
   - `GET /api/models/available-models` - List available ML models
   - `POST /api/models/evaluate/<model_name>` - Evaluate specific model

### Running the HTML Frontend

1. **Start the HTML frontend server**
   ```bash
   python run_html_frontend.py
   ```
   - Frontend will be available at `http://localhost:8000`
   - This serves the static HTML/CSS/JS files

2. **Using the Web Interface**
   - Open `http://localhost:8000` in your browser
   - Fill in the product specifications form
   - Click "Get Recommendations" to see AI-powered suggestions
   - View detailed results including confidence scores and similar materials

### Running the Streamlit Frontend (Alternative)

1. **Start the Streamlit frontend**
   ```bash
   python run_frontend.py
   ```
   The frontend will be available at `http://localhost:8501`

2. **Using the Interface**
   - Select product category from dropdown
   - Adjust sliders for fragility, sustainability, and other parameters
   - Click "Get Recommendations" to see AI-powered suggestions
   - View confidence scores and similar materials

### Quick Demo

**Run the complete demo** (starts both backend and frontend):
```bash
python demo_html_frontend.py
```
This will:
- Check if the backend is running
- Start the HTML frontend server
- Open your browser automatically
- Show demo instructions

## Project Structure

```
src/
├── api/                 # Flask API application
│   ├── app.py          # Main Flask app
│   └── routes/         # API route handlers
├── frontend/           # Streamlit web interface
│   └── app.py          # Main Streamlit app
├── models/             # Machine learning models
├── etl/                # Data processing pipelines
├── data/               # Data loading and processing
├── nlp/                # Natural language processing
├── utils/              # Utility functions
└── config/             # Configuration management

data/                   # Data files and datasets
models/                 # Trained model artifacts
mlruns/                 # MLflow experiment tracking
```

## Development

### Training New Models

```bash
# Train a specific model
python -m src.models.train_random_forest

# Evaluate models
python -m src.api.routes.models_routes  # Use the evaluate endpoint
```

### Adding New Features

1. Update the feature engineering in `src/etl/feature_engineering.py`
2. Retrain models with new features
3. Update API endpoints if needed
4. Update frontend interface

## API Documentation

### Authentication

Most API endpoints require an API key in the `X-API-Key` header.

### Product Recommendation

**Endpoint:** `POST /api/product/recommend-materials`

**Request Body:**
```json
{
  "product_category": "electronics",
  "fragility_score": 0.8,
  "sustainability_priority": 0.9,
  "material_cost": 45.0,
  "durability_requirement": 0.7,
  "max_packaging_cost": 100.0,
  "innovation_level": 4
}
```

**Response:**
```json
{
  "status": "success",
  "product_score": 0.85,
  "recommendation_level": "Highly Recommended",
  "recommended": true,
  "confidence_score": 0.85,
  "recommendations": [
    {
      "material": "Recycled Cardboard",
      "confidence": 85.0,
      "reason": "High recommendation confidence with excellent sustainability profile"
    }
  ],
  "similar_materials": [...],
  "model_info": {...}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.