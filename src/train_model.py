import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
from imblearn.over_sampling import SMOTE


class DoubleClap:
    def __init__(self):
        self.scaler = None
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = None
        
    def load_data(self, features_path, labels_path):
        """Load the feature data"""
        X = np.load(features_path)
        y = np.load(labels_path)
        
        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """
        Advanced preprocessing with scaling and balancing
        """
        # Remove features with zero variance
        feature_var = np.var(X, axis=0)
        valid_features = feature_var > 1e-8
        X_filtered = X[:, valid_features]
        
        print(f"Removed {np.sum(~valid_features)} zero-variance features")
        print(f"Remaining features: {X_filtered.shape[1]}")
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features using RobustScaler (less sensitive to outliers)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, valid_features
    
    def train_individual_models(self, X_train, y_train):
        """
        Train multiple individual models
        """
        # Calculate class weights for imbalanced data
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # Define models with hyperparameter tuning
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True, class_weight='balanced'),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['adaptive'],
                    'activation': ['relu', 'tanh']
                }
            }
        }
        
        # Train and tune each model
        for name, config in models_config.items():
            print(f"\\nTraining {name}...")
            
            # Use SMOTE to balance the dataset within cross-validation
            smote = SMOTE(random_state=42)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            # Apply SMOTE and train
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            grid_search.fit(X_balanced, y_balanced)
            
            self.models[name] = grid_search.best_estimator_
            print(f"Best {name} score: {grid_search.best_score_:.4f}")
            print(f"Best {name} params: {grid_search.best_params_}")
    
    def create_ensemble(self, X_train, y_train):
        """
        Create ensemble model from individual models
        """
        # Apply SMOTE for ensemble training
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        # Create voting classifier
        estimators = [(name, model) for name, model in self.models.items()]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability averages
        )
        
        print("\\nTraining ensemble model...")
        self.ensemble_model.fit(X_balanced, y_balanced)
        
        # Feature importance (from Random Forest)
        if 'random_forest' in self.models:
            self.feature_importance = self.models['random_forest'].feature_importances_
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models on test set
        """
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            print(f"\\n{name.upper()} Results:")
            print(f"Accuracy: {results[name]['accuracy']:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
        
        # Evaluate ensemble
        if self.ensemble_model:
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            y_prob_ensemble = self.ensemble_model.predict_proba(X_test)[:, 1]
            
            results['ensemble'] = {
                'accuracy': accuracy_score(y_test, y_pred_ensemble),
                'predictions': y_pred_ensemble,
                'probabilities': y_prob_ensemble
            }
            
            print("\\nENSEMBLE Results:")
            print(f"Accuracy: {results['ensemble']['accuracy']:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred_ensemble))
        
        return results
    
    def save_models(self, model_dir='data/processed'):
        """
        Save all trained models
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, f'{model_dir}/scaler.joblib')
        
        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, f'{model_dir}/{name}.joblib')
        
        # Save ensemble
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, f'{model_dir}/model.joblib')
        
        print(f"Models saved to {model_dir}")
    
    def predict_with_confidence(self, X, confidence_threshold=0.7):
        """
        Make predictions with confidence scores
        """
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.ensemble_model.predict_proba(X_scaled)
        predictions = self.ensemble_model.predict(X_scaled)
        
        # Only return positive predictions with high confidence
        confident_predictions = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            max_prob = np.max(prob)
            if pred == 1 and max_prob >= confidence_threshold:
                confident_predictions.append((i, pred, max_prob))
            elif pred == 0:  # Always trust negative predictions
                confident_predictions.append((i, pred, max_prob))
            else:
                confident_predictions.append((i, 0, max_prob))  # Low confidence -> negative
        
        return confident_predictions


def main():
    # Initialize the model
    model = DoubleClap()
    
    # Load features
    print("Loading features...")
    X, y = model.load_data('data/processed/X.npy', 'data/processed/y.npy')
    
    # Preprocess data
    print("\\nPreprocessing data...")
    X_train, X_test, y_train, y_test, valid_features = model.preprocess_data(X, y)
    
    # Train individual models
    print("\\nTraining individual models...")
    model.train_individual_models(X_train, y_train)
    
    # Create ensemble
    print("\\nCreating ensemble...")
    model.create_ensemble(X_train, y_train)
    
    # Evaluate models
    print("\\nEvaluating models...")
    results = model.evaluate_models(X_test, y_test)
    
    # Save models
    model.save_models()
    
    print("\\nTraining completed! Models saved.")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
