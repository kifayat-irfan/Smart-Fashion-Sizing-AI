import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

class FashionDataPreprocessor:
    """
    Handles the cleaning, outlier treatment, and feature engineering 
    of 3D Anthropometry data for fashion size prediction.
    """
    
    def __init__(self):
        # IterativeImputer models each feature as a function of others.
        # This is superior to mean/median for body measurements where 
        # height and weight are strongly correlated.
        self.imputer = IterativeImputer(max_iter=10, random_state=42)
        self.scaler = StandardScaler()
        self.feature_cols = ['Height', 'Weight', 'Waist', 'Hip', 'Arm_Length']

    def handle_missing_values(self, df):
        """
        Fills missing values using Multivariate Imputation by Chained Equations (MICE).
        """
        print("Handling missing values via Iterative Imputation...")
        df_numeric = df[self.feature_cols]
        imputed_data = self.imputer.fit_transform(df_numeric)
        
        # Replace original columns with imputed values
        df[self.feature_cols] = imputed_data
        return df

    def handle_outliers_iqr(self, df):
        """
        Capping outliers using the Interquartile Range (IQR) method.
        Why: Anthropometric data often contains extreme anomalies (errors) 
        that can skew the Neural Network's weights.
        """
        print("Capping outliers using IQR...")
        for col in self.feature_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip values to the bounds instead of dropping them to preserve dataset size
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    def engineer_features(self, df):
        """
        Creates domain-specific fashion and health metrics.
        - BMI: General body mass indicator.
        - WHR: Primary indicator for Body Shape (Apple vs Pear).
        - Arm_Height_Ratio: Helps determine Fit (Sleeve length/Proportion).
        """
        print("Engineering derived fashion features...")
        
        # BMI = weight (kg) / [height (m)]^2 
        # Assuming height is in cm, convert to meters
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
        
        # Waist-to-Hip Ratio (WHR) - Critical for Pear/Apple/Hourglass logic
        df['WHR'] = df['Waist'] / df['Hip']
        
        # Arm-to-Height Ratio - Critical for Regular vs Slim fit logic
        df['Arm_Height_Ratio'] = df['Arm_Length'] / df['Height']
        
        return df

    def scale_features(self, df, training=True):
        """
        Standardizes features to mean=0 and variance=1.
        Neural Networks converge significantly faster when inputs are on the same scale.
        """
        # We include the engineered features in scaling
        all_numeric_cols = self.feature_cols + ['BMI', 'WHR', 'Arm_Height_Ratio']
        
        if training:
            scaled_values = self.scaler.fit_transform(df[all_numeric_cols])
        else:
            scaled_values = self.scaler.transform(df[all_numeric_cols])
            
        scaled_df = pd.DataFrame(scaled_values, columns=all_numeric_cols, index=df.index)
        
        # Concatenate back with non-numeric columns (like Gender)
        non_numeric = df.drop(columns=all_numeric_cols)
        return pd.concat([non_numeric, scaled_df], axis=1)

    def pipeline(self, df, training=True):
        """
        Orchestrates the full preprocessing workflow.
        """
        df = df.copy()
        df = self.handle_missing_values(df)
        df = self.handle_outliers_iqr(df)
        df = self.engineer_features(df)
        df = self.scale_features(df, training=training)
        return df
