.PHONY: all setup train serve test clean docker-build docker-run

# Define python interpreter
PYTHON := python3

# Default target
all: setup train serve

# Create virtual environment and install dependencies
setup:
	@echo "--- Setting up virtual environment and installing dependencies ---"
	$(PYTHON) -m venv .venv
	@. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "--- Downloading Kaggle dataset ---"
	@bash ./data/download_data.sh
	@echo "Setup complete. Activate the venv with: source .venv/bin/activate"

# Train all models
train:
	@echo "--- Training all models (baseline, xgb, nn) ---"
	@. .venv/bin/activate && $(PYTHON) src/train.py --model all

# Train XGBoost model only
train-xgb:
	@echo "--- Training XGBoost model ---"
	@. .venv/bin/activate && $(PYTHON) src/train.py --model xgb

# Serve the Streamlit application
serve:
	@echo "--- Starting Streamlit app. Navigate to http://localhost:8501 ---"
	@. .venv/bin/activate && streamlit run app/streamlit_app.py

# Run tests
test:
	@echo "--- Running tests with pytest ---"
	@. .venv/bin/activate && pytest

# Build the Docker image
docker-build:
	@echo "--- Building Docker image: house-price-app ---"
	@docker build -t house-price-app .

# Run the Docker container
docker-run:
	@echo "--- Running Docker container. App will be at http://localhost:8501 ---"
	@docker run -p 8501:8501 --rm house-price-app

# Clean up generated files
clean:
	@echo "--- Cleaning up generated files ---"
	@rm -rf .venv __pycache__ */__pycache__ .pytest_cache .coverage
	@rm -rf models/*.joblib models/*.json models/*.keras models/*.csv
	@rm -rf data/raw data/processed
	@echo "Cleanup complete."