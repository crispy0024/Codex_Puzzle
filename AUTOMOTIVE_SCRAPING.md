# Automotive Data Scraping and Machine Learning

This module provides functionality to scrape automotive data from autotrader.com and perform machine learning analysis on the collected data.

## Features

### Web Scraping
- **AutoTrader Scraper**: Extracts vehicle listings from autotrader.com
- **VehicleData**: Data class for storing vehicle information
- **Sample Data**: Demonstration data for testing purposes

### Machine Learning Analysis
- **Price Prediction**: Train models to predict vehicle prices based on features
- **Vehicle Categorization**: Classify vehicles into categories (luxury, economy, truck, etc.)
- **Market Analysis**: Analyze trends in the automotive data

## API Endpoints

All scraping endpoints are prefixed with `/scraping/`:

### Data Management
- `GET /scraping/sample_data` - Load sample automotive data
- `POST /scraping/scrape_vehicles` - Scrape vehicles (currently returns sample data)
- `GET /scraping/stored_data` - Retrieve all stored vehicle data
- `DELETE /scraping/clear_data` - Clear all stored data
- `GET /scraping/status` - Get system status

### Machine Learning
- `POST /scraping/train_price_model` - Train price prediction model
- `POST /scraping/train_category_model` - Train vehicle categorization model
- `POST /scraping/predict_price` - Predict vehicle price
- `GET /scraping/market_analysis` - Get market trend analysis

## Usage Examples

### 1. Load Sample Data
```bash
curl http://localhost:5000/scraping/sample_data
```

### 2. Train Price Prediction Model
```bash
curl -X POST http://localhost:5000/scraping/train_price_model
```

### 3. Predict Vehicle Price
```bash
curl -X POST http://localhost:5000/scraping/predict_price \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Toyota",
    "model": "Camry", 
    "year": 2023,
    "mileage": 10000,
    "fuel_type": "Gasoline",
    "transmission": "Automatic"
  }'
```

### 4. Get Market Analysis
```bash
curl http://localhost:5000/scraping/market_analysis
```

## Data Structure

### VehicleData Fields
- `make`: Vehicle manufacturer (e.g., "Toyota", "Honda")
- `model`: Vehicle model (e.g., "Camry", "Civic")
- `year`: Model year
- `price`: Listing price in USD
- `mileage`: Odometer reading
- `location`: Geographic location
- `fuel_type`: Fuel type (Gasoline, Electric, etc.)
- `transmission`: Transmission type (Automatic, Manual)
- `exterior_color`: Exterior paint color
- `drivetrain`: Drive system (FWD, AWD, etc.)
- `engine`: Engine description

## Machine Learning Models

### Price Prediction
Uses Random Forest Regression to predict vehicle prices based on:
- Year, mileage, make, fuel type, transmission, drivetrain

### Vehicle Categorization
Classifies vehicles into categories:
- **LUXURY**: BMW, Mercedes, Audi, Lexus, etc.
- **ECONOMY**: Toyota, Honda, Nissan, Hyundai, etc.
- **TRUCK**: Ford, Chevrolet, GMC, Ram, etc.
- **ELECTRIC**: Tesla
- **OTHER**: All other makes

## Note on Web Scraping

The current implementation provides sample data for demonstration purposes. Real web scraping of autotrader.com should:

1. Respect the website's robots.txt and terms of service
2. Implement proper rate limiting and delays between requests
3. Handle dynamic content and anti-bot measures
4. Use appropriate user agents and headers
5. Consider using official APIs when available

## Integration with Existing System

The automotive scraping functionality integrates seamlessly with the existing puzzle application:

- Uses the same FastAPI framework and server structure
- Follows existing code patterns and conventions
- Maintains separate modules for different functionality
- Includes comprehensive tests following existing test patterns

## Testing

Run tests for the scraping functionality:

```bash
python -m pytest tests/test_scraping.py -v
```

All tests include:
- Unit tests for scraper and ML analyzer components
- API endpoint tests
- Integration tests with sample data
- Error handling tests