import pandas as pd
import numpy as np

class FashionLabelGenerator:
    """
    Synthesizes ground-truth labels for Size, Fit, and Body Shape 
    based on anthropometric industry standards.
    """
    
    def __init__(self):
        # Define BMI thresholds for Fit categorization
        self.fit_thresholds = {
            'slim': 22.0,
            'regular': 27.0 
            # < 22: Slim, 22-27: Regular, > 27: Relaxed
        }

    def assign_size(self, df):
        """
        Predicts Size (S, M, L, XL) based on a weighted combination 
        of Weight and BMI quantiles.
        """
        print("Generating Size labels...")
        
        q1 = df['Weight'].quantile(0.25)
        q2 = df['Weight'].quantile(0.50)
        q3 = df['Weight'].quantile(0.75)

        def size_logic(row):
            w = row['Weight']
            if w <= q1: return 'S'
            elif w <= q2: return 'M'
            elif w <= q3: return 'L'
            else: return 'XL'

        return df.apply(size_logic, axis=1)

    def assign_fit(self, df):
        """
        Predicts Fit (Slim, Regular, Relaxed).
        """
        print("Generating Fit labels...")
        
        def fit_logic(row):
            bmi = row['BMI']
            if bmi < self.fit_thresholds['slim']:
                return 'Slim'
            elif bmi < self.fit_thresholds['regular']:
                return 'Regular'
            else:
                return 'Relaxed'

        return df.apply(fit_logic, axis=1)

    def assign_body_shape(self, df):
        """
        Predicts Body Shape (Pear, Apple, Hourglass, Rectangle).
        """
        print("Generating Body Shape labels...")

        def shape_logic(row):
            whr = row['WHR']
            
            # --- FIX START ---
            # We wrap Gender in str() to prevent crashes if the value is NaN (float).
            # We use 'in' instead of '==' to handle cases like " female" or "Female "
            gender = str(row['Gender']).lower()
            # --- FIX END ---
            
            if 'female' in gender:
                if whr < 0.7:
                    # Hourglass typically has a more extreme waist pinch 
                    # but balanced proportions.
                    return 'Hourglass' if row['Waist'] < df['Waist'].quantile(0.3) else 'Pear'
                elif whr > 0.85:
                    return 'Apple'
                else:
                    return 'Rectangle'
            else: # Male or other
                if whr > 0.9:
                    return 'Apple'
                elif whr < 0.8:
                    return 'Pear'
                else:
                    return 'Rectangle'

        return df.apply(shape_logic, axis=1)

    def generate_all_labels(self, df):
        """
        Orchestrates the labeling process and adds targets to the dataframe.
        """
        df = df.copy()
        df['Target_Size'] = self.assign_size(df)
        df['Target_Fit'] = self.assign_fit(df)
        df['Target_Shape'] = self.assign_body_shape(df)
        
        return df
