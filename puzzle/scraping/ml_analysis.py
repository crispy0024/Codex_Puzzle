"""Machine learning analysis for automotive data"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from typing import List, Dict, Any, Optional, Tuple
import logging

from .autotrader_scraper import VehicleData

logger = logging.getLogger(__name__)


class AutomotiveMLAnalyzer:
    """Machine learning analyzer for automotive data"""
    
    def __init__(self):
        """Initialize the ML analyzer"""
        self.price_model: Optional[RandomForestRegressor] = None
        self.category_model: Optional[RandomForestClassifier] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns = []
        
    def prepare_data(self, vehicles: List[VehicleData]) -> pd.DataFrame:
        """
        Convert list of VehicleData to pandas DataFrame and prepare for ML
        
        Args:
            vehicles: List of VehicleData objects
            
        Returns:
            Prepared DataFrame
        """
        # Convert to DataFrame
        data = [vehicle.to_dict() for vehicle in vehicles]
        df = pd.DataFrame(data)
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['make', 'model', 'year'])
        
        # Clean and standardize data
        df = self._clean_data(df)
        
        return df
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the automotive data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Clean numeric columns
        numeric_columns = ['year', 'price', 'mileage']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Remove outliers
        if 'year' in df.columns:
            df = df[(df['year'] >= 1990) & (df['year'] <= 2024)]
            
        if 'price' in df.columns:
            df = df[(df['price'] > 1000) & (df['price'] < 200000)]
            
        if 'mileage' in df.columns:
            df = df[df['mileage'] >= 0]
            
        # Standardize string columns
        string_columns = ['make', 'model', 'fuel_type', 'transmission', 'drivetrain']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                
        return df
        
    def train_price_prediction_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a model to predict vehicle prices
        
        Args:
            df: Prepared DataFrame with vehicle data
            
        Returns:
            Dictionary with training results
        """
        try:
            # Filter out rows without price data
            df_price = df.dropna(subset=['price']).copy()
            
            if len(df_price) < 5:
                return {"error": "Insufficient data for price prediction (need at least 5 samples)"}
                
            # Prepare features
            X = self._prepare_features(df_price)
            y = df_price['price'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.price_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.price_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Handle NaN values for JSON serialization
            if np.isnan(r2):
                r2 = 0.0
            if np.isnan(mse):
                mse = 0.0
            
            # Feature importance
            feature_importance = {}
            for i, col in enumerate(self.feature_columns):
                importance = float(self.price_model.feature_importances_[i])
                if np.isnan(importance):
                    importance = 0.0
                feature_importance[col] = importance
            
            results = {
                "model_type": "price_prediction",
                "samples_used": len(df_price),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2_score": float(r2),
                "feature_importance": feature_importance
            }
            
            logger.info(f"Price prediction model trained with R² = {r2:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {e}")
            return {"error": str(e)}
            
    def train_vehicle_category_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train a model to classify vehicles by category/type
        
        Args:
            df: Prepared DataFrame with vehicle data
            
        Returns:
            Dictionary with training results
        """
        try:
            # Create categories based on make/model
            df_cat = df.dropna(subset=['make']).copy()
            df_cat['category'] = df_cat['make'].apply(self._categorize_vehicle)
            
            if len(df_cat) < 5:
                return {"error": "Insufficient data for category classification (need at least 5 samples)"}
                
            # Prepare features
            X = self._prepare_features(df_cat)
            y = df_cat['category'].values
            
            # Encode target labels
            if 'category' not in self.label_encoders:
                self.label_encoders['category'] = LabelEncoder()
                
            y_encoded = self.label_encoders['category'].fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train model
            self.category_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.category_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.category_model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            # Feature importance
            feature_importance = {}
            for i, col in enumerate(self.feature_columns):
                importance = float(self.category_model.feature_importances_[i])
                if np.isnan(importance):
                    importance = 0.0
                feature_importance[col] = importance
            
            results = {
                "model_type": "category_classification",
                "samples_used": len(df_cat),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": float(accuracy),
                "categories": list(self.label_encoders['category'].classes_),
                "feature_importance": feature_importance
            }
            
            logger.info(f"Category classification model trained with accuracy = {accuracy:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training category classification model: {e}")
            return {"error": str(e)}
            
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for ML models
        
        Args:
            df: DataFrame with vehicle data
            
        Returns:
            Feature matrix as numpy array
        """
        features = []
        feature_names = []
        
        # Numeric features
        numeric_features = ['year', 'mileage']
        for col in numeric_features:
            if col in df.columns:
                values = df[col].fillna(df[col].median() if not df[col].empty else 0)
            else:
                values = pd.Series([0] * len(df))
            features.append(values.values.reshape(-1, 1))
            feature_names.append(col)
                
        # Categorical features (encode as numeric)
        categorical_features = ['make', 'fuel_type', 'transmission', 'drivetrain']
        for col in categorical_features:
            if col in df.columns:
                values = df[col].fillna('UNKNOWN')
            else:
                values = pd.Series(['UNKNOWN'] * len(df))
                
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                
            # Handle missing values and new categories
            unique_values = values.unique()
            
            # Fit encoder if not already fitted
            if not hasattr(self.label_encoders[col], 'classes_'):
                self.label_encoders[col].fit(unique_values)
            else:
                # Handle new categories
                existing_classes = set(self.label_encoders[col].classes_)
                new_classes = set(unique_values) - existing_classes
                if new_classes:
                    all_classes = list(existing_classes) + list(new_classes)
                    self.label_encoders[col].classes_ = np.array(all_classes)
                    
            encoded = self.label_encoders[col].transform(values)
            features.append(encoded.reshape(-1, 1))
            feature_names.append(col)
                
        if features:
            X = np.hstack(features)
            
            # Scale features
            if self.scaler is None:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
                
            # Store feature names if not already set
            if not self.feature_columns:
                self.feature_columns = feature_names
                
            return X
        else:
            return np.array([]).reshape(len(df), 0)
            
    def _categorize_vehicle(self, make: str) -> str:
        """
        Categorize vehicle by make into broader categories
        
        Args:
            make: Vehicle make
            
        Returns:
            Vehicle category
        """
        make = str(make).upper()
        
        luxury_brands = ['BMW', 'MERCEDES', 'AUDI', 'LEXUS', 'ACURA', 'INFINITI', 'CADILLAC']
        truck_brands = ['FORD', 'CHEVROLET', 'GMC', 'RAM', 'DODGE']
        economy_brands = ['TOYOTA', 'HONDA', 'NISSAN', 'HYUNDAI', 'KIA']
        electric_brands = ['TESLA']
        
        if make in luxury_brands:
            return 'LUXURY'
        elif make in truck_brands:
            return 'TRUCK'
        elif make in economy_brands:
            return 'ECONOMY'
        elif make in electric_brands:
            return 'ELECTRIC'
        else:
            return 'OTHER'
            
    def predict_price(self, vehicle_data: Dict[str, Any]) -> Optional[float]:
        """
        Predict price for a vehicle
        
        Args:
            vehicle_data: Dictionary with vehicle features
            
        Returns:
            Predicted price or None if prediction fails
        """
        if self.price_model is None:
            return None
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame([vehicle_data])
            df = self._clean_data(df)
            
            # Fill missing features with defaults
            for col in ['year', 'mileage', 'make', 'fuel_type', 'transmission', 'drivetrain']:
                if col not in df.columns:
                    if col in ['year', 'mileage']:
                        df[col] = 0
                    else:
                        df[col] = 'UNKNOWN'
            
            # Prepare features using the same method as training
            X = self._prepare_features(df)
            
            if X.shape[1] == 0:
                return None
                
            # Predict
            prediction = self.price_model.predict(X)[0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            return None
            
    def analyze_market_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market trends in the automotive data
        
        Args:
            df: DataFrame with vehicle data
            
        Returns:
            Dictionary with market analysis
        """
        try:
            analysis = {}
            
            # Price analysis by make
            if 'make' in df.columns and 'price' in df.columns:
                price_by_make = df.groupby('make')['price'].agg(['mean', 'median', 'count']).round(2)
                # Convert to simple dict for JSON serialization
                analysis['price_by_make'] = {}
                for make in price_by_make.index:
                    analysis['price_by_make'][make] = {
                        'mean': float(price_by_make.loc[make, 'mean']),
                        'median': float(price_by_make.loc[make, 'median']),
                        'count': int(price_by_make.loc[make, 'count'])
                    }
                
            # Year analysis
            if 'year' in df.columns:
                year_stats = df['year'].describe()
                analysis['year_distribution'] = {}
                for k, v in year_stats.to_dict().items():
                    analysis['year_distribution'][k] = float(v) if not np.isnan(v) else 0.0
                
            # Mileage analysis
            if 'mileage' in df.columns:
                mileage_stats = df['mileage'].describe()
                analysis['mileage_distribution'] = {}
                for k, v in mileage_stats.to_dict().items():
                    analysis['mileage_distribution'][k] = float(v) if not np.isnan(v) else 0.0
                
            # Most common combinations
            if all(col in df.columns for col in ['make', 'model']):
                top_models = df.groupby(['make', 'model']).size().sort_values(ascending=False).head(10)
                analysis['popular_models'] = {}
                for (make, model), count in top_models.items():
                    key = f"{make} {model}"
                    analysis['popular_models'][key] = int(count)
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {e}")
            return {"error": str(e)}