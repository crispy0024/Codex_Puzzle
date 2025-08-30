"""AutoTrader.com web scraper for vehicle data extraction"""

import time
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import logging
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class VehicleData:
    """Data class for vehicle information scraped from AutoTrader"""
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    price: Optional[int] = None
    mileage: Optional[int] = None
    location: Optional[str] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    exterior_color: Optional[str] = None
    interior_color: Optional[str] = None
    drivetrain: Optional[str] = None
    engine: Optional[str] = None
    listing_url: Optional[str] = None
    dealer_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class AutoTraderScraper:
    """Web scraper for AutoTrader.com vehicle listings"""
    
    def __init__(self, delay_between_requests: float = 2.0):
        """
        Initialize the scraper
        
        Args:
            delay_between_requests: Delay in seconds between requests to be respectful
        """
        self.delay = delay_between_requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def search_vehicles(self, make: str = "", model: str = "", max_results: int = 20) -> List[VehicleData]:
        """
        Search for vehicles on AutoTrader.com
        
        Args:
            make: Vehicle make (e.g., "Toyota", "Honda")
            model: Vehicle model (e.g., "Camry", "Civic")
            max_results: Maximum number of results to return
            
        Returns:
            List of VehicleData objects
        """
        vehicles = []
        
        try:
            # Build search URL
            base_url = "https://www.autotrader.com/cars-for-sale/all-cars"
            params = {
                'searchRadius': '25',
                'location': '10001',  # New York City default
                'isNewSearch': 'true',
                'showAccelerateBanner': 'false',
                'sortBy': 'relevance'
            }
            
            if make:
                params['makeCodeList'] = make.upper()
            if model:
                params['modelCodeList'] = model.upper()
                
            # Make request to search page
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for vehicle listings in the page
            vehicle_cards = soup.find_all('div', {'data-cmp': 'inventoryListing'})
            
            if not vehicle_cards:
                # Try alternative selectors that might work with current AutoTrader layout
                vehicle_cards = soup.find_all('div', class_=re.compile(r'.*listing.*|.*inventory.*'))
                
            logger.info(f"Found {len(vehicle_cards)} vehicle cards on page")
            
            for i, card in enumerate(vehicle_cards[:max_results]):
                if i > 0:
                    time.sleep(self.delay)
                    
                vehicle_data = self._extract_vehicle_data(card, response.url)
                if vehicle_data:
                    vehicles.append(vehicle_data)
                    
        except requests.RequestException as e:
            logger.error(f"Error fetching search results: {e}")
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            
        return vehicles
        
    def _extract_vehicle_data(self, card_element, page_url: str) -> Optional[VehicleData]:
        """
        Extract vehicle data from a listing card element
        
        Args:
            card_element: BeautifulSoup element containing vehicle listing
            page_url: URL of the page being scraped
            
        Returns:
            VehicleData object or None if extraction fails
        """
        try:
            vehicle = VehicleData()
            
            # Extract title which often contains make, model, year
            title_elem = card_element.find('h2') or card_element.find('h3') or card_element.find(string=re.compile(r'\d{4}'))
            if title_elem:
                title_text = title_elem.get_text(strip=True) if hasattr(title_elem, 'get_text') else str(title_elem)
                vehicle.make, vehicle.model, vehicle.year = self._parse_title(title_text)
                
            # Extract price
            price_elem = card_element.find(string=re.compile(r'\$[\d,]+')) or card_element.find(text=re.compile(r'\$[\d,]+'))
            if price_elem:
                price_text = str(price_elem)
                price_match = re.search(r'\$?([\d,]+)', price_text)
                if price_match:
                    try:
                        vehicle.price = int(price_match.group(1).replace(',', ''))
                    except ValueError:
                        pass
                        
            # Extract mileage
            mileage_elem = card_element.find(string=re.compile(r'[\d,]+\s*mil', re.I))
            if mileage_elem:
                mileage_match = re.search(r'([\d,]+)', str(mileage_elem))
                if mileage_match:
                    try:
                        vehicle.mileage = int(mileage_match.group(1).replace(',', ''))
                    except ValueError:
                        pass
                        
            # Extract location
            location_elem = card_element.find(string=re.compile(r'[A-Z]{2}\s*\d{5}')) or card_element.find(text=re.compile(r'[A-Z]{2}'))
            if location_elem:
                vehicle.location = str(location_elem).strip()
                
            # Try to find a link to the full listing
            link_elem = card_element.find('a', href=True)
            if link_elem:
                href = link_elem['href']
                if href.startswith('/'):
                    vehicle.listing_url = f"https://www.autotrader.com{href}"
                elif href.startswith('http'):
                    vehicle.listing_url = href
                    
            # Only return if we extracted at least basic info
            if vehicle.make or vehicle.model or vehicle.price:
                return vehicle
                
        except Exception as e:
            logger.error(f"Error extracting vehicle data: {e}")
            
        return None
        
    def _parse_title(self, title: str) -> tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Parse vehicle title to extract make, model, and year
        
        Args:
            title: Title string to parse
            
        Returns:
            Tuple of (make, model, year)
        """
        make = model = None
        year = None
        
        # Extract year (4-digit number, typically 19xx or 20xx)
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        if year_match:
            try:
                year = int(year_match.group())
            except ValueError:
                pass
                
        # Remove year from title for make/model extraction
        title_no_year = re.sub(r'\b(19|20)\d{2}\b', '', title).strip()
        
        # Split remaining title and try to identify make/model
        words = title_no_year.split()
        if len(words) >= 2:
            make = words[0]
            model = words[1]
        elif len(words) == 1:
            make = words[0]
            
        return make, model, year
        
    def get_sample_data(self) -> List[VehicleData]:
        """
        Get sample vehicle data for testing/demonstration purposes
        
        Returns:
            List of sample VehicleData objects
        """
        sample_data = [
            VehicleData(
                make="Toyota",
                model="Camry",
                year=2022,
                price=25000,
                mileage=15000,
                location="New York, NY",
                fuel_type="Gasoline",
                transmission="Automatic",
                exterior_color="Silver",
                drivetrain="FWD",
                engine="2.5L 4-Cylinder"
            ),
            VehicleData(
                make="Honda",
                model="Civic",
                year=2021,
                price=22000,
                mileage=20000,
                location="Los Angeles, CA",
                fuel_type="Gasoline",
                transmission="Manual",
                exterior_color="Blue",
                drivetrain="FWD",
                engine="2.0L 4-Cylinder"
            ),
            VehicleData(
                make="Ford",
                model="F-150",
                year=2023,
                price=35000,
                mileage=5000,
                location="Dallas, TX",
                fuel_type="Gasoline",
                transmission="Automatic",
                exterior_color="Red",
                drivetrain="4WD",
                engine="3.5L V6"
            ),
            VehicleData(
                make="Tesla",
                model="Model 3",
                year=2022,
                price=45000,
                mileage=8000,
                location="San Francisco, CA",
                fuel_type="Electric",
                transmission="Automatic",
                exterior_color="White",
                drivetrain="RWD",
                engine="Electric Motor"
            ),
            VehicleData(
                make="BMW",
                model="X5",
                year=2023,
                price=55000,
                mileage=3000,
                location="Seattle, WA",
                fuel_type="Gasoline",
                transmission="Automatic",
                exterior_color="Black",
                drivetrain="AWD",
                engine="3.0L 6-Cylinder"
            )
        ]
        return sample_data