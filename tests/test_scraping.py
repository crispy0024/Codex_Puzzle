"""Tests for automotive data scraping functionality"""

import pytest
import json
from fastapi.testclient import TestClient
from puzzle.api import create_app
from puzzle.scraping import AutoTraderScraper, VehicleData, AutomotiveMLAnalyzer


@pytest.fixture
def client():
    """Create test client"""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_vehicle_data():
    """Create sample vehicle data for testing"""
    return [
        VehicleData(
            make="Toyota",
            model="Camry",
            year=2022,
            price=25000,
            mileage=15000,
            location="New York, NY",
            fuel_type="Gasoline",
            transmission="Automatic"
        ),
        VehicleData(
            make="Honda", 
            model="Civic",
            year=2021,
            price=22000,
            mileage=20000,
            location="Los Angeles, CA",
            fuel_type="Gasoline",
            transmission="Manual"
        )
    ]


def test_autotrader_scraper_initialization():
    """Test that AutoTrader scraper initializes properly"""
    scraper = AutoTraderScraper()
    assert scraper is not None
    assert scraper.delay == 2.0
    assert scraper.session is not None


def test_vehicle_data_creation():
    """Test VehicleData dataclass functionality"""
    vehicle = VehicleData(
        make="Toyota",
        model="Camry", 
        year=2022,
        price=25000
    )
    assert vehicle.make == "Toyota"
    assert vehicle.model == "Camry"
    assert vehicle.year == 2022
    assert vehicle.price == 25000
    
    # Test to_dict conversion
    vehicle_dict = vehicle.to_dict()
    assert vehicle_dict["make"] == "Toyota"
    assert vehicle_dict["model"] == "Camry"


def test_scraper_get_sample_data():
    """Test that scraper can generate sample data"""
    scraper = AutoTraderScraper()
    sample_data = scraper.get_sample_data()
    
    assert len(sample_data) > 0
    assert all(isinstance(vehicle, VehicleData) for vehicle in sample_data)
    assert all(vehicle.make is not None for vehicle in sample_data)


def test_ml_analyzer_initialization():
    """Test ML analyzer initialization"""
    analyzer = AutomotiveMLAnalyzer()
    assert analyzer is not None
    assert analyzer.price_model is None
    assert analyzer.category_model is None


def test_ml_analyzer_prepare_data(sample_vehicle_data):
    """Test data preparation for ML analysis"""
    analyzer = AutomotiveMLAnalyzer()
    df = analyzer.prepare_data(sample_vehicle_data)
    
    assert len(df) == 2
    assert 'make' in df.columns
    assert 'model' in df.columns
    assert 'year' in df.columns
    assert 'price' in df.columns


def test_scraping_sample_data_endpoint(client):
    """Test sample data endpoint"""
    response = client.get("/scraping/sample_data")
    assert response.status_code == 200
    
    data = response.json()
    assert "vehicles" in data
    assert "count" in data
    assert data["count"] > 0
    assert len(data["vehicles"]) == data["count"]


def test_scraping_status_endpoint(client):
    """Test status endpoint"""
    response = client.get("/scraping/status")
    assert response.status_code == 200
    
    data = response.json()
    assert "stored_vehicles" in data
    assert "price_model_trained" in data
    assert "category_model_trained" in data
    assert "available_endpoints" in data


def test_scrape_vehicles_endpoint(client):
    """Test vehicle scraping endpoint"""
    request_data = {
        "make": "Toyota",
        "model": "Camry", 
        "max_results": 5
    }
    
    response = client.post("/scraping/scrape_vehicles", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "vehicles" in data
    assert "count" in data
    assert "search_params" in data


def test_stored_data_endpoint(client):
    """Test stored data retrieval endpoint"""
    # First add some data
    client.get("/scraping/sample_data")
    
    # Then retrieve it
    response = client.get("/scraping/stored_data")
    assert response.status_code == 200
    
    data = response.json()
    assert "vehicles" in data
    assert "count" in data


def test_clear_data_endpoint(client):
    """Test data clearing endpoint"""
    # First add some data
    client.get("/scraping/sample_data")
    
    # Clear it
    response = client.delete("/scraping/clear_data")
    assert response.status_code == 200
    
    data = response.json()
    assert "count" in data
    assert data["count"] == 0


def test_train_price_model_endpoint(client):
    """Test price model training endpoint"""
    # First add sample data
    client.get("/scraping/sample_data")
    
    # Train model
    response = client.post("/scraping/train_price_model")
    assert response.status_code == 200
    
    data = response.json()
    assert "success" in data
    assert data["success"] == True
    assert "model_results" in data


def test_train_category_model_endpoint(client):
    """Test category model training endpoint"""
    # First add sample data
    client.get("/scraping/sample_data")
    
    # Train model
    response = client.post("/scraping/train_category_model")
    assert response.status_code == 200
    
    data = response.json()
    assert "success" in data
    assert data["success"] == True
    assert "model_results" in data


def test_predict_price_endpoint(client):
    """Test price prediction endpoint"""
    # First add sample data and train model
    client.get("/scraping/sample_data")
    client.post("/scraping/train_price_model")
    
    # Make prediction
    prediction_data = {
        "make": "Toyota",
        "model": "Camry",
        "year": 2022,
        "mileage": 15000
    }
    
    response = client.post("/scraping/predict_price", json=prediction_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price" in data
    assert "input_data" in data
    assert isinstance(data["predicted_price"], (int, float))


def test_market_analysis_endpoint(client):
    """Test market analysis endpoint"""
    # First add sample data
    client.get("/scraping/sample_data")
    
    # Get analysis
    response = client.get("/scraping/market_analysis")
    assert response.status_code == 200
    
    data = response.json()
    assert "analysis" in data
    assert "data_count" in data


def test_ml_analyzer_market_trends(sample_vehicle_data):
    """Test market trends analysis"""
    analyzer = AutomotiveMLAnalyzer()
    df = analyzer.prepare_data(sample_vehicle_data)
    
    analysis = analyzer.analyze_market_trends(df)
    assert isinstance(analysis, dict)
    
    # Should have some analysis results
    assert len(analysis) > 0


def test_ml_analyzer_categorize_vehicle():
    """Test vehicle categorization logic"""
    analyzer = AutomotiveMLAnalyzer()
    
    assert analyzer._categorize_vehicle("TOYOTA") == "ECONOMY"
    assert analyzer._categorize_vehicle("BMW") == "LUXURY"
    assert analyzer._categorize_vehicle("FORD") == "TRUCK"
    assert analyzer._categorize_vehicle("TESLA") == "ELECTRIC"
    assert analyzer._categorize_vehicle("UNKNOWN") == "OTHER"


def test_title_parsing():
    """Test title parsing functionality"""
    scraper = AutoTraderScraper()
    
    # Test various title formats
    make, model, year = scraper._parse_title("2022 Toyota Camry LE")
    assert year == 2022
    assert make == "Toyota"
    assert model == "Camry"
    
    make, model, year = scraper._parse_title("Honda Civic 2021")
    assert year == 2021
    assert make == "Honda"
    assert model == "Civic"


def test_insufficient_data_error(client):
    """Test error handling with insufficient data"""
    # Clear any existing data
    client.delete("/scraping/clear_data")
    
    # Try to train model with no data
    response = client.post("/scraping/train_price_model")
    assert response.status_code == 400
    
    data = response.json()
    assert "error" in data