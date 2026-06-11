import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np
from src.model import FashionMultiOutputNet

class FashionHyperTuner:
    def __init__(self, X_train, y_size, y_fit, y_shape, X_test, y_size_test, y_fit_test, y_shape_test):
        self.X_train = X_train
        self.y_size = y_size
        self.y_fit = y_fit
        self.y_shape = y_shape
        self.X_test = X_test
        self.y_size_test = y_size_test
        self.y_fit_test = y_fit_test
        self.y_shape_test = y_shape_test

    def evaluate_config(self, config):
        # 1. Setup Data
        train_ds = TensorDataset(
            torch.tensor(self.X_train), 
            torch.tensor(self.y_size), 
            torch.tensor(self.y_fit), 
            torch.tensor(self.y_shape)
        )
        loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

        # 2. Initialize Model using the new flexible constructor
        model = FashionMultiOutputNet(
            input_dim=self.X_train.shape[1], 
            h1=config['hidden_1'], 
            h2=config['hidden_2'], 
            size_out=len(np.unique(self.y_size)), 
            fit_out=len(np.unique(self.y_fit)), 
            shape_out=len(np.unique(self.y_shape)),
            dropout_rate=config['dropout']
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        # 3. Train
        model.train()
        for epoch in range(20): 
            for bx, bs, bf, bsh in loader:
                optimizer.zero_grad()
                os, of, osh = model(bx)
                loss = criterion(os, bs) + criterion(of, bf) + criterion(osh, bsh)
                loss.backward()
                optimizer.step()

        # 4. Evaluate using Macro F1
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(self.X_test).float()
            p_s, p_f, p_sh = model(X_test_tensor)
            p_s = torch.argmax(p_s, dim=1).numpy()
            p_f = torch.argmax(p_f, dim=1).numpy()
            p_sh = torch.argmax(p_sh, dim=1).numpy()

        f1_s = f1_score(self.y_size_test, p_s, average='macro')
        f1_f = f1_score(self.y_fit_test, p_f, average='macro')
        f1_sh = f1_score(self.y_shape_test, p_sh, average='macro')
        
        return (f1_s + f1_f + f1_sh) / 3

    def tune(self, search_space):
        best_score = -1
        best_config = None
        import itertools
        keys, values = zip(*search_space.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"Testing {len(combinations)} different configurations...")
        for i, config in enumerate(combinations):
            score = self.evaluate_config(config)
            print(f"Config {i+1}/{len(combinations)} | Score: {score:.4f} | Params: {config}")
            if score > best_score:
                best_score = score
                best_config = config
        
        return best_config, best_score
