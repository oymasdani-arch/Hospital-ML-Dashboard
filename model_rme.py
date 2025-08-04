# model_rme.py
# ==========================================
# Script Training Model untuk Prediksi Lama Rawat Inap
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 🔧 Configuration
# ==========================================
class Config:
    DATA_PATH = "data/data_rme_dummy_baru.csv"
    MODEL_DIR = "model"
    OUTPUT_DIR = "output"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

# ==========================================
# 📁 Setup Directories
# ==========================================
def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print("✅ Directories created successfully!")

# ==========================================
# 📊 Data Loading and Preprocessing
# ==========================================
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("📥 Loading dataset...")
    
    try:
        df = pd.read_csv(Config.DATA_PATH)
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: File {Config.DATA_PATH} not found!")
        print("Please make sure the data file exists in the correct path.")
        return None, None, None
    
    # Display basic information about the dataset
    print("\n📊 Dataset Info:")
    print(f"- Total rows: {len(df):,}")
    print(f"- Total columns: {df.shape[1]}")
    print(f"- Missing values: {df.isnull().sum().sum()}")
    
    # Display column names
    print(f"\n📋 Columns: {list(df.columns)}")
    
    # Target variable statistics
    print(f"\n🎯 Target Variable (Lama_Rawat) Stats:")
    print(f"- Mean: {df['Lama_Rawat'].mean():.2f} days")
    print(f"- Median: {df['Lama_Rawat'].median():.2f} days")
    print(f"- Min: {df['Lama_Rawat'].min()} days")
    print(f"- Max: {df['Lama_Rawat'].max()} days")
    print(f"- Std: {df['Lama_Rawat'].std():.2f} days")
    
    # Prepare features and target
    print("\n🔧 Preparing features and target...")
    X = df.drop(columns=["Lama_Rawat", "Cepat_Sembuh", "Inisial_Nama"])
    y = df["Lama_Rawat"]
    
    # Encode categorical variables
    print("🔄 Encoding categorical variables...")
    X_encoded = pd.get_dummies(X, drop_first=False)
    
    print(f"✅ Features prepared! Shape after encoding: {X_encoded.shape}")
    print(f"📋 Total features: {len(X_encoded.columns)}")
    
    return df, X_encoded, y

# ==========================================
# 🤖 Model Definitions
# ==========================================
def get_models():
    """Define all models with their configurations"""
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {}  # No hyperparameters to tune for basic LinearRegression
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(random_state=Config.RANDOM_STATE),
            "params": {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=Config.RANDOM_STATE),
            "params": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=Config.RANDOM_STATE),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=Config.RANDOM_STATE, eval_metric='rmse'),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        }
    }
    return models

# ==========================================
# 📊 Model Evaluation
# ==========================================
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    return metrics, y_pred

# ==========================================
# 🚀 Model Training with Hyperparameter Tuning
# ==========================================
def train_models(X_train, X_test, y_train, y_test, models):
    """Train all models with hyperparameter tuning"""
    results = {}
    best_models = {}
    
    print("🚀 Starting model training with hyperparameter tuning...")
    print("=" * 60)
    
    for name, model_config in models.items():
        print(f"\n🔄 Training {name}...")
        
        start_time = datetime.now()
        
        if model_config["params"]:  # If hyperparameters are defined
            print(f"   🔍 Performing GridSearch for {name}...")
            grid_search = GridSearchCV(
                model_config["model"],
                model_config["params"],
                cv=Config.CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"   ✅ Best parameters: {grid_search.best_params_}")
            print(f"   📊 Best CV score: {grid_search.best_score_:.4f}")
        else:
            # For models without hyperparameter tuning (like basic LinearRegression)
            best_model = model_config["model"]
            best_model.fit(X_train, y_train)
        
        # Evaluate the model
        metrics, y_pred = evaluate_model(best_model, X_test, y_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=Config.CV_FOLDS, scoring='r2')
        
        # Store results
        results[name] = {
            'model': best_model,
            'metrics': metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred
        }
        
        best_models[name] = best_model
        
        # Training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ⏱️  Training time: {training_time:.2f} seconds")
        print(f"   📊 Test R²: {metrics['R²']:.4f}")
        print(f"   📊 Test RMSE: {metrics['RMSE']:.4f}")
        print(f"   📊 CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        print(f"   ✅ {name} training completed!")
    
    return results, best_models

# ==========================================
# 💾 Save Models and Results
# ==========================================
def save_models_and_results(best_models, X_encoded, results):
    """Save trained models and results"""
    print("\n💾 Saving models and results...")
    
    # Save feature columns
    joblib.dump(X_encoded.columns.tolist(), f"{Config.MODEL_DIR}/model_columns.pkl")
    print("✅ Model columns saved!")
    
    # Save individual models
    for name, model in best_models.items():
        model_path = f"{Config.MODEL_DIR}/model_{name}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ {name} model saved to {model_path}")
    
    # Save results summary
    results_summary = []
    for name, result in results.items():
        summary = {
            'Model': name,
            'MAE': result['metrics']['MAE'],
            'MSE': result['metrics']['MSE'],
            'RMSE': result['metrics']['RMSE'],
            'R²': result['metrics']['R²'],
            'MAPE': result['metrics']['MAPE'],
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std']
        }
        results_summary.append(summary)
    
    results_df = pd.DataFrame(results_summary)
    results_path = f"{Config.OUTPUT_DIR}/model_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✅ Results summary saved to {results_path}")
    
    return results_df

# ==========================================
# 📊 Generate Evaluation Report
# ==========================================
def generate_evaluation_report(results_df, results):
    """Generate comprehensive evaluation report"""
    print("\n📊 Generating evaluation report...")
    
    # Find best model
    best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
    best_r2_score = results_df['R²'].max()
    
    print("=" * 60)
    print("📊 MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
    print(f"   📊 R² Score: {best_r2_score:.4f}")
    print(f"   📊 RMSE: {results[best_model_name]['metrics']['RMSE']:.4f}")
    print(f"   📊 MAE: {results[best_model_name]['metrics']['MAE']:.4f}")
    print(f"   📊 MAPE: {results[best_model_name]['metrics']['MAPE']:.2f}%")
    
    print(f"\n📋 ALL MODELS PERFORMANCE:")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<18} | R²: {row['R²']:.4f} | RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f}")
    
    print("\n🎯 PERFORMANCE RANKING (by R² Score):")
    print("-" * 40)
    ranked_models = results_df.sort_values('R²', ascending=False)
    for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
        print(f"{i}. {row['Model']:<18} (R²: {row['R²']:.4f})")
    
    # Model recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    if best_r2_score > 0.8:
        print("✅ Excellent model performance! Ready for deployment.")
    elif best_r2_score > 0.6:
        print("✅ Good model performance. Consider feature engineering for improvement.")
    elif best_r2_score > 0.4:
        print("⚠️  Moderate performance. More data or feature engineering recommended.")
    else:
        print("❌ Poor performance. Review data quality and feature selection.")
    
    return best_model_name

# ==========================================
# 📈 Visualization
# ==========================================
def create_visualizations(results_df, results):
    """Create and save visualizations"""
    print("\n📈 Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. R² Score comparison
    axes[0, 0].bar(results_df['Model'], results_df['R²'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. RMSE comparison
    axes[0, 1].bar(results_df['Model'], results_df['RMSE'], color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. MAE comparison
    axes[1, 0].bar(results_df['Model'], results_df['MAE'], color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('MAE Comparison')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. MAPE comparison
    axes[1, 1].bar(results_df['Model'], results_df['MAPE'], color='gold', alpha=0.7)
    axes[1, 1].set_title('MAPE Comparison (%)')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = f"{Config.OUTPUT_DIR}/model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {plot_path}")
    plt.close()

# ==========================================
# 🔍 Feature Importance Analysis
# ==========================================
def analyze_feature_importance(best_models, X_encoded, model_name):
    """Analyze and save feature importance for tree-based models"""
    print(f"\n🔍 Analyzing feature importance for {model_name}...")
    
    if model_name in ['RandomForest', 'DecisionTree', 'GradientBoosting', 'XGBoost']:
        model = best_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importance_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Save top 20 features
            top_features = importance_df.head(20)
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['Importance'], alpha=0.7)
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            importance_plot_path = f"{Config.OUTPUT_DIR}/feature_importance_{model_name}.png"
            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save to CSV
            importance_csv_path = f"{Config.OUTPUT_DIR}/feature_importance_{model_name}.csv"
            importance_df.to_csv(importance_csv_path, index=False)
            
            print(f"✅ Feature importance analysis saved!")
            print(f"   📊 Plot: {importance_plot_path}")
            print(f"   📋 CSV: {importance_csv_path}")
            
            print(f"\n🔝 Top 5 Most Important Features:")
            for i, (_, row) in enumerate(top_features.head().iterrows(), 1):
                print(f"   {i}. {row['Feature']}: {row['Importance']:.4f}")

# ==========================================
# 🎯 Main Training Pipeline
# ==========================================
def main():
    """Main training pipeline"""
    print("🏥 PREDIKSI LAMA RAWAT INAP - MODEL TRAINING")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Load and preprocess data
    df, X_encoded, y = load_and_preprocess_data()
    if df is None:
        return
    
    # Split data
    print(f"\n🔄 Splitting data (test size: {Config.TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE
    )
    print(f"✅ Data split completed!")
    print(f"   📊 Training set: {X_train.shape[0]:,} samples")
    print(f"   📊 Test set: {X_test.shape[0]:,} samples")
    
    # Get models
    models = get_models()
    print(f"\n🤖 Models to train: {list(models.keys())}")
    
    # Train models
    results, best_models = train_models(X_train, X_test, y_train, y_test, models)
    
    # Save models and results
    results_df = save_models_and_results(best_models, X_encoded, results)
    
    # Generate report
    best_model_name = generate_evaluation_report(results_df, results)
    
    # Create visualizations
    create_visualizations(results_df, results)
    
    # Feature importance analysis
    analyze_feature_importance(best_models, X_encoded, best_model_name)
    
    print("\n" + "=" * 60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Models saved in: {Config.MODEL_DIR}/")
    print(f"📊 Results saved in: {Config.OUTPUT_DIR}/")
    print("\n💡 Next steps:")
    print("   1. Run the dashboard: streamlit run dashboard.py")
    print("   2. Review the evaluation results")
    print("   3. Test the best performing model")
    print("=" * 60)

if __name__ == "__main__":
    main()