import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Import our custom modular components
from src.preprocessing import FashionDataPreprocessor
from src.fashion_logic import FashionLabelGenerator
from src.model import FashionMultiOutputNet
from src.evaluation import FashionEvaluator
from src.tuner import FashionHyperTuner

def main():
    # ==========================================
    # 1. Setup and Data Loading
    # ==========================================
    print("--- Step 1: Loading CAESAR dataset ---")
    try:
        df = pd.read_csv('data/caesar.csv')
    except FileNotFoundError:
        print("Error: caesar.csv not found in data/ folder.")
        return

    column_mapping = {
        'Height': 'Height', 'Weight': 'Weight', 'Waist': 'Waist', 
        'Hip': 'Hip', 'Arm_Length': 'Arm_Length', 'Gender': 'Gender'
    }
    df = df.rename(columns=column_mapping)
    required_cols = list(column_mapping.values())
    df = df[required_cols]
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    # ==========================================
    # 2. Preprocessing Pipeline
    # ==========================================
    print("\n--- Step 2: Preprocessing ---")
    preprocessor = FashionDataPreprocessor()
    df_processed = preprocessor.pipeline(df)

    # ==========================================
    # 3. Fashion Logic (Generating Ground Truth)
    # ==========================================
    print("\n--- Step 3: Generating Fashion Labels ---")
    label_gen = FashionLabelGenerator()
    df_labeled = label_gen.generate_all_labels(df_processed)

    # ==========================================
    # 4. Categorical Encoding
    # ==========================================
    print("\n--- Step 4: Encoding Variables ---")
    le_gender = LabelEncoder()
    df_labeled['Gender'] = le_gender.fit_transform(df_labeled['Gender'].astype(str))

    le_size = LabelEncoder()
    le_fit = LabelEncoder()
    le_shape = LabelEncoder()

    df_labeled['Target_Size'] = le_size.fit_transform(df_labeled['Target_Size'])
    df_labeled['Target_Fit'] = le_fit.fit_transform(df_labeled['Target_Fit'])
    df_labeled['Target_Shape'] = le_shape.fit_transform(df_labeled['Target_Shape'])

    # ==========================================
    # 5. PyTorch Dataset Preparation
    # ==========================================
    X = df_labeled.drop(columns=['Target_Size', 'Target_Fit', 'Target_Shape']).values.astype(np.float32)
    y_size = df_labeled['Target_Size'].values.astype(np.int64)
    y_fit = df_labeled['Target_Fit'].values.astype(np.int64)
    y_shape = df_labeled['Target_Shape'].values.astype(np.int64)

    X_train, X_test, y_size_train, y_size_test, y_fit_train, y_fit_test, y_shape_train, y_shape_test = \
        train_test_split(X, y_size, y_fit, y_shape, test_size=0.2, random_state=42)

    # ==========================================
    # 5.5 Hyperparameter Tuning
    # ==========================================
    print("\n--- Step 5.5: Hyperparameter Tuning ---")
    search_space = {
        'lr': [0.001, 0.0001],
        'hidden_1': [64, 128],
        'hidden_2': [32, 64],
        'dropout': [0.2, 0.3],
        'batch_size': [32, 64]
    }
    
    tuner = FashionHyperTuner(X_train, y_size_train, y_fit_train, y_shape_train, 
                              X_test, y_size_test, y_fit_test, y_shape_test)
    
    best_params, best_score = tuner.tune(search_space)
    print(f"\nBest Config Found: {best_params} with Score: {best_score:.4f}")

    # ==========================================
    # 6. Model Initialization (Using Best Params)
    # ==========================================
    print("\n--- Step 6: Initializing Final Model with Best Params ---")
    model = FashionMultiOutputNet(
        input_dim=X.shape[1], 
        h1=best_params['hidden_1'], 
        h2=best_params['hidden_2'], 
        size_out=len(le_size.classes_), 
        fit_out=len(le_fit.classes_), 
        shape_out=len(le_shape.classes_),
        dropout_rate=best_params['dropout']
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

    # Create final DataLoader using the best batch size
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_size_train), 
        torch.tensor(y_fit_train), torch.tensor(y_shape_train)
    )
    train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True)

    # ==========================================
    # 7. Training Loop
    # ==========================================
    print("\nStarting Final Training...")
    model.train()
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_size, batch_fit, batch_shape in train_loader:
            optimizer.zero_grad()
            out_size, out_fit, out_shape = model(batch_x)
            loss = criterion(out_size, batch_size) + \
                   criterion(out_fit, batch_fit) + \
                   criterion(out_shape, batch_shape)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Combined Loss: {total_loss/len(train_loader):.4f}")

    # ==========================================
    # 8. Professional Evaluation
    # ==========================================
    print("\n--- Step 7: Final Evaluation ---")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).float()
        p_size, p_fit, p_shape = model(X_test_tensor)
        p_size = torch.argmax(p_size, dim=1).numpy()
        p_fit = torch.argmax(p_fit, dim=1).numpy()
        p_shape = torch.argmax(p_shape, dim=1).numpy()

    evaluator = FashionEvaluator()
    evaluator.print_metrics(y_size_test, p_size, le_size.classes_, "Size")
    evaluator.plot_confusion_matrix(y_size_test, p_size, le_size.classes_, "Size")
    evaluator.print_metrics(y_fit_test, p_fit, le_fit.classes_, "Fit")
    evaluator.plot_confusion_matrix(y_fit_test, p_fit, le_fit.classes_, "Fit")
    evaluator.print_metrics(y_shape_test, p_shape, le_shape.classes_, "Body Shape")
    evaluator.plot_confusion_matrix(y_shape_test, p_shape, le_shape.classes_, "Body Shape")

if __name__ == "__main__":
    main()
