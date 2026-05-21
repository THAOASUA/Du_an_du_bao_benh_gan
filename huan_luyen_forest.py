import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Duong dan du lieu
path = r"F:\tài liệu\New folder\Liver_disease_data.csv"

try:
    print("--- TRAINING RANDOM FOREST MODEL ---")
    
    # Doc du lieu
    df = pd.read_csv(path, encoding='latin1')
    print(f"Loaded data: {df.shape}")
    
    # Tien xu ly
    cols_to_drop = ['PatientID', 'Diagnosis', 'ID', 'patientid']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    
    feature_names = [c for c in df.columns if c not in existing_drops]
    X = df[feature_names].values
    y = df['Diagnosis'].values
    
    print(f"Using {X.shape[1]} features")
    print(f"Dataset size: {len(df)} samples")
    
    # Chia du lieu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tim tham so tot nhat (GridSearch)
    print("\nSearching for best parameters...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_*100:.2f}%")
    
    # Train with best parameters
    best_rf = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest set accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(best_rf, "rf_model.pkl")
    print(f"\nModel saved: {os.path.abspath('rf_model.pkl')}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 important features:")
    print(importance_df.head())
    
    # Save feature importance
    importance_df.to_csv("rf_feature_importance.csv", index=False)
    
    print("\n" + "="*50)
    print("COMPLETED! Random Forest model is ready!")
    print("="*50)

except FileNotFoundError:
    print(f"ERROR: Cannot find file at {path}")
    print("Please check your CSV file path!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()