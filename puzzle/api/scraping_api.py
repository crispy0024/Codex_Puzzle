"""FastAPI endpoints for automotive data scraping and analysis"""

import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..scraping import AutoTraderScraper, VehicleData, AutomotiveMLAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances
_scraper: Optional[AutoTraderScraper] = None
_analyzer: Optional[AutomotiveMLAnalyzer] = None
_vehicle_data: List[VehicleData] = []


def get_scraper() -> AutoTraderScraper:
    """Get or create AutoTrader scraper instance"""
    global _scraper
    if _scraper is None:
        _scraper = AutoTraderScraper(delay_between_requests=1.0)
    return _scraper


def get_analyzer() -> AutomotiveMLAnalyzer:
    """Get or create ML analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = AutomotiveMLAnalyzer()
    return _analyzer


class ScrapeRequest(BaseModel):
    """Request model for scraping vehicles"""
    make: str = ""
    model: str = ""
    max_results: int = 20


class PredictionRequest(BaseModel):
    """Request model for price prediction"""
    make: str
    model: str
    year: int
    mileage: Optional[int] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None


@router.get("/scraping/sample_data")
async def get_sample_data():
    """Get sample automotive data for testing"""
    try:
        scraper = get_scraper()
        sample_vehicles = scraper.get_sample_data()
        
        # Store sample data globally
        global _vehicle_data
        _vehicle_data.clear()  # Clear existing data first
        _vehicle_data.extend(sample_vehicles)
        
        return {
            "vehicles": [vehicle.to_dict() for vehicle in sample_vehicles],
            "count": len(sample_vehicles),
            "message": "Sample automotive data loaded"
        }
    except Exception as e:
        logger.error(f"Error getting sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scraping/scrape_vehicles")
async def scrape_vehicles(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Scrape vehicle data from AutoTrader.com
    
    Note: This endpoint provides sample data for demonstration.
    Real scraping would require careful consideration of the website's terms of service.
    """
    try:
        scraper = get_scraper()
        
        # For demonstration, return sample data instead of actual scraping
        # Real implementation would call: vehicles = scraper.search_vehicles(request.make, request.model, request.max_results)
        vehicles = scraper.get_sample_data()
        
        # Filter sample data based on request
        if request.make or request.model:
            filtered_vehicles = []
            for vehicle in vehicles:
                make_match = not request.make or (vehicle.make and request.make.lower() in vehicle.make.lower())
                model_match = not request.model or (vehicle.model and request.model.lower() in vehicle.model.lower())
                if make_match and model_match:
                    filtered_vehicles.append(vehicle)
            vehicles = filtered_vehicles[:request.max_results]
        else:
            vehicles = vehicles[:request.max_results]
        
        # Store scraped data globally
        global _vehicle_data
        _vehicle_data.extend(vehicles)
        
        return {
            "vehicles": [vehicle.to_dict() for vehicle in vehicles],
            "count": len(vehicles),
            "search_params": {
                "make": request.make,
                "model": request.model,
                "max_results": request.max_results
            },
            "message": f"Found {len(vehicles)} vehicles (using sample data for demonstration)"
        }
        
    except Exception as e:
        logger.error(f"Error scraping vehicles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scraping/stored_data")
async def get_stored_data():
    """Get all stored vehicle data"""
    try:
        global _vehicle_data
        return {
            "vehicles": [vehicle.to_dict() for vehicle in _vehicle_data],
            "count": len(_vehicle_data),
            "message": "Retrieved stored vehicle data"
        }
    except Exception as e:
        logger.error(f"Error getting stored data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/scraping/clear_data")
async def clear_stored_data():
    """Clear all stored vehicle data"""
    try:
        global _vehicle_data
        count = len(_vehicle_data)
        _vehicle_data = []
        return {
            "message": f"Cleared {count} vehicle records",
            "count": 0
        }
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scraping/train_price_model")
async def train_price_model():
    """Train machine learning model for price prediction"""
    try:
        global _vehicle_data
        
        if len(_vehicle_data) < 5:
            return JSONResponse(
                status_code=400,
                content={"error": "Need at least 5 vehicle records to train model"}
            )
        
        analyzer = get_analyzer()
        df = analyzer.prepare_data(_vehicle_data)
        
        if len(df) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid data found for training"}
            )
        
        results = analyzer.train_price_prediction_model(df)
        
        if "error" in results:
            return JSONResponse(status_code=400, content=results)
        
        return {
            "success": True,
            "model_results": results,
            "message": "Price prediction model trained successfully"
        }
        
    except Exception as e:
        logger.error(f"Error training price model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scraping/train_category_model")
async def train_category_model():
    """Train machine learning model for vehicle categorization"""
    try:
        global _vehicle_data
        
        if len(_vehicle_data) < 5:
            return JSONResponse(
                status_code=400,
                content={"error": "Need at least 5 vehicle records to train model"}
            )
        
        analyzer = get_analyzer()
        df = analyzer.prepare_data(_vehicle_data)
        
        if len(df) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid data found for training"}
            )
        
        results = analyzer.train_vehicle_category_model(df)
        
        if "error" in results:
            return JSONResponse(status_code=400, content=results)
        
        return {
            "success": True,
            "model_results": results,
            "message": "Vehicle category model trained successfully"
        }
        
    except Exception as e:
        logger.error(f"Error training category model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scraping/predict_price")
async def predict_price(request: PredictionRequest):
    """Predict vehicle price using trained ML model"""
    try:
        analyzer = get_analyzer()
        
        if analyzer.price_model is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Price prediction model not trained. Train model first."}
            )
        
        vehicle_data = {
            "make": request.make,
            "model": request.model,
            "year": request.year,
            "mileage": request.mileage,
            "fuel_type": request.fuel_type,
            "transmission": request.transmission
        }
        
        predicted_price = analyzer.predict_price(vehicle_data)
        
        if predicted_price is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not predict price with given data"}
            )
        
        return {
            "predicted_price": round(predicted_price, 2),
            "input_data": vehicle_data,
            "message": "Price predicted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error predicting price: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scraping/market_analysis")
async def get_market_analysis():
    """Get market trend analysis of stored vehicle data"""
    try:
        global _vehicle_data
        
        if len(_vehicle_data) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No vehicle data available for analysis"}
            )
        
        analyzer = get_analyzer()
        df = analyzer.prepare_data(_vehicle_data)
        
        if len(df) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid data found for analysis"}
            )
        
        analysis = analyzer.analyze_market_trends(df)
        
        return {
            "analysis": analysis,
            "data_count": len(_vehicle_data),
            "valid_data_count": len(df),
            "message": "Market analysis completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scraping/status")
async def get_scraping_status():
    """Get status of scraping system"""
    try:
        global _vehicle_data
        analyzer = get_analyzer()
        
        return {
            "stored_vehicles": len(_vehicle_data),
            "price_model_trained": analyzer.price_model is not None,
            "category_model_trained": analyzer.category_model is not None,
            "available_endpoints": [
                "/scraping/sample_data",
                "/scraping/scrape_vehicles",
                "/scraping/stored_data", 
                "/scraping/clear_data",
                "/scraping/train_price_model",
                "/scraping/train_category_model",
                "/scraping/predict_price",
                "/scraping/market_analysis",
                "/scraping/status"
            ],
            "message": "Automotive scraping system operational"
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))