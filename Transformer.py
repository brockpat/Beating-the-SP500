# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:30:15 2025

@author: patri
"""

#%% Libraries

path = "C:/Users/patri/Desktop/ML/"

#DataFrame Libraries
import pandas as pd
import sqlite3

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Scientifiy Libraries
# Scientific Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import random

import os
os.chdir(path + "Code/")
import General_Functions as GF

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch"
)

#%% Seeds
# --- 1. Define Seeding Function for Reproducibility ---
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#%% Preliminaries & Data

# =================================================
#               Preliminaries
# =================================================

#Database
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_processed.db")
db_Predictions = sqlite3.connect(database=path +"Data/Predictions.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")

#Get Settings & signals
settings = GF.get_settings()
signals = GF.get_signals()
feat_cols = signals[0] + signals[1] #Continuous and Categorical Features

# =================================================
#                Read in Data
# =================================================

#Target
target_col = 'tr_m_sp500_ld1'

#Load Data
df, signal_months, trading_month_start, feat_cols, \
    window_size, validation_size  \
        = GF.load_retpred_data(db_signals = JKP_Factors,       #Database with signals
                               db_sp500 = SP500_Constituents,  #Database with S&P500 Constituents
                               settings = settings,            #General settings
                               target = target_col,            #Prediction target
                               rank_signals = True,            #Use Ranks
                               trade_start = '2004-01-31',     #First trading date
                               trade_end = '2024-12-31',       #Last trading date
                               continuous_signals = signals[0],#List of continuous signals
                               fill_missing_values = True,     #Fill missing values 
                               CRSP = False,                   #Estimation on CRSP universe or S&P 500 universe
                               )


# =================================================
#                Universe Size
# =================================================

# Maximum no. of stocks in each cross-section. 
MAX_UNIVERSE_SIZE = df.groupby('eom')['id'].count().max()
"""
This has a lookahead-bias, but it is purely for computational efficiency so that
the ''context window'' only is as big as it has to be. This has no effect whatsoever
on the return predictions being fully OOS.
"""

# ================================================
#           Rolling Window Parameters
# ================================================

#Window Size and Validation Periods for rolling window
window_size = settings['rolling_window']['window_size']
validation_size = settings['rolling_window']['validation_periods'] 
test_size = settings['rolling_window']['test_size'] #Periods until hyperparameters are re-tuned. Fine-tuning is done monthly

#Trading Dates
trading_dates = signal_months[trading_month_start:]

# ================================================
#        Model Type (Requires Manual Naming)
# ================================================
model_name = "TransformerSet_Dropout010"
target_type = "ZscoreTarget"
file_end = f"SP500UniverseFL_RankFeatures_RollingWindow_win{window_size}_val{validation_size}_test{test_size}"
prediction_name = "ret_pred_Zscore"

model_save_path = path + "Models/Transformer_Set/"

#%%
colab_package = {
    'df': df,
    'feat_cols': feat_cols,
    'target_col': target_col,
    'trading_dates': trading_dates, 
    'signal_months': signal_months,
    'trading_month_start': trading_month_start,
    # Settings params
    'window_size': window_size,
    'validation_size': validation_size,
    'test_size': test_size,
    'MAX_UNIVERSE_SIZE': MAX_UNIVERSE_SIZE
}

# 2. Define export path (Save it directly to your Google Drive folder if you have Google Drive for Desktop installed, otherwise save to Desktop)
export_path = path + "Data/ready_for_colab.pkl"

# 3. Save as Pickle
# protocol=4 ensures compatibility across different Python versions
import pickle
with open(export_path, 'wb') as f:
    pickle.dump(colab_package, f, protocol=4)

print(f"Successfully saved package to {export_path}")
print(f"File size: {os.path.getsize(export_path) / (1024*1024):.2f} MB")

#%% PyTorch Model & Utilities

# =====================================
#  Define the Transformer Architecture
# =====================================
class StockTransformer(nn.Module):
    def __init__(self, d_feat, 
                 d_model=64,            # No. of columns fed into the attention block 
                 nhead=4,               # No. of Attention Heads 
                 num_layers=2,          # 2 Transformer blocks 
                 dim_feedforward=256,   # No. of Neurons in the Feed-Forward Layer
                 dropout=0.05):         #Before: 0.2
        super(StockTransformer, self).__init__()
        
        # 1. Feature Embedding: Project stock features to d_model (e.g. 128)
        # Using a Linear layer acts as a learnable embedding
        self.embedding = nn.Linear(d_feat, d_model)
        self.layer_norm_input = nn.LayerNorm(d_model)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=True, #Input tensor shape: (batch, seq_length, n_features)
            norm_first=True # Pre-LN 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Head
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1) # Regressing to stock return
        )
        
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask=None):
        # src shape: [Batch, Max_Stocks, d_feat]
        
        x = self.embedding(src)
        x = self.layer_norm_input(x)
        
        # Transformer pass
        # src_key_padding_mask is True for padded positions
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Decode
        output = self.decoder(x)
        return output.squeeze(-1) # Return [Batch, Max_Stocks]

def prepare_tensor_data(df_slice, feat_cols, target_col, max_len, device):
    """
    Converts a pandas DataFrame slice (one or multiple months) into PyTorch tensors
    with padding for cross-sectional alignment.
    
    Each Batch is one cross-section. This is important, because the Transformer
    here only operates strictly within a batch since the prediction is made
    using cross-section t for next month t+1.
    
    CAUTION: Depending on the size it can be more efficient to ONLY move the batch
    that is currently trained to the GPU. In the default setting, the entire training
    data is moved to the GPU.
    """
    unique_dates = df_slice['eom'].unique()
    unique_dates = unique_dates[unique_dates.argsort()]
    
    batch_X = []
    batch_y = []
    batch_mask = []
    batch_ids = [] # To track which prediction belongs to which stock
    
    for date in unique_dates:
        # Filter data for this month
        mask_date = df_slice['eom'] == date
        sub_df = df_slice[mask_date]
        
        # Extract features and targets
        vals = sub_df[feat_cols].values
        targs = sub_df[target_col].values
        ids = sub_df[['id', 'eom']].copy()
        
        num_stocks = len(vals)
        
        # PADDING LOGIC
        # We need shape (max_len, features)
        # Create placeholders filled with zeros
        padded_X = np.zeros((max_len, len(feat_cols)), dtype=np.float32)
        padded_y = np.zeros((max_len,), dtype=np.float32)
        
        # Mask: False = Real Data, True = Padding (PyTorch convention)
        padded_mask = np.ones((max_len,), dtype=bool) 
        
        # Fill real data
        # Note: We just stack them at the beginning. Since it's a set transformer
        # without positional encoding, order doesn't strictly matter.
        padded_X[:num_stocks, :] = vals
        padded_y[:num_stocks] = targs
        padded_mask[:num_stocks] = False
        
        batch_X.append(padded_X)
        batch_y.append(padded_y)
        batch_mask.append(padded_mask)
        batch_ids.append(ids)
        
    # Stack into tensors 
    X_tensor = torch.tensor(np.array(batch_X), dtype=torch.float32).to(device) #(Batch, max_len, feat_cols)
    y_tensor = torch.tensor(np.array(batch_y), dtype=torch.float32).to(device) #(Batch, max_len)
    mask_tensor = torch.tensor(np.array(batch_mask), dtype=torch.bool).to(device) #(Batch,max_len)
    
    return X_tensor, y_tensor, mask_tensor, batch_ids

def train_step(model, optimizer, criterion, X, y, mask):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    preds = model(X, src_key_padding_mask=mask)
    
    # We must only calculate loss on NON-PADDED elements
    # ~mask selects valid elements (False in mask means Valid)
    valid_preds = preds[~mask]
    valid_y = y[~mask]
    
    loss = criterion(valid_preds, valid_y)
    loss.backward()
    
    # Gradient clipping is crucial for Transformers
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

#%% Training & Testing
        
#Empty list to store the predictions
predictions = []

# Track how many periods since last hyperparameter tuning
months_since_tune = test_size   # start so that the first iteration triggers tuning

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
#  Global Model Hyperparameters
# ==============================

# Define how many months (cross-sections) constitute one gradient update step
BATCH_SIZE = 16
# Base learning rates
BASE_LR = 1e-3
# Hyperparameters to be found
saved_best_epochs = None      # Will store the optimal base training epochs
saved_initial_state = None    # Store the starting weights

#Loop over dates (Rolling Window)
for trade_idx, date in enumerate(trading_dates, start=trading_month_start):
    
    # =======================================
    # Get the Data for the Rolling Regression
    # =======================================
    
    #Rolling Window Dates
    start_idx = max(0, trade_idx - (window_size + validation_size+1))
    window_months = signal_months[start_idx:trade_idx-1] #!!!End is not inclusive!!!
    #Note: we can only use signals up to t-2 since we are predicting next period's targelt. Else-wise: Look-ahead bias (data leakage)
    #Seed for reproducability
    current_seed = int(42 + trade_idx)
    set_seed(current_seed)
    
    # =================================================
    #    Hyperparameter tuning every test_size months
    # =================================================
    if months_since_tune >= test_size:
    
        print(f"Re-tuning hyperparameters at date {date}")
        
        # Split into train / val months
        train_months = window_months[:window_size]
        val_months   = window_months[window_size:]
        
        # Training & Validation Data
        df_train = df[df['eom'].isin(train_months)]
        df_val   = df[df['eom'].isin(val_months)]
        
        
        # --- A. Train Base Model on 10 Years ---
        # Initialize a FRESH model
        temp_model = StockTransformer(d_feat=len(feat_cols)).to(device)
        # Save the initialisation
        saved_initial_state = temp_model.state_dict()
        
        temp_optimizer = optim.AdamW(temp_model.parameters(), lr=BASE_LR, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # ---- Prepare Data & Loader ----
        X_train, y_train, mask_train, ids_train = prepare_tensor_data(
            df_train, feat_cols, target_col, MAX_UNIVERSE_SIZE, device)
        # The resulting X_train shape is (Total_Months, MAX_UNIVERSE_SIZE, Features)
        
        # Create an Index Tensor
        indices_train = torch.arange(len(X_train))
        
        # TensorDataset indexes the *first* dimension. 
        #   So dataset[0] returns the entire cross-section for the first month.
        #   train_dataset.tensors is the tuple of tensors
        train_dataset = TensorDataset(X_train, y_train, mask_train, indices_train) 
        train_loader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, # Number of Months per Update
                                  shuffle=True, # Randomizes WHICH months form a batch, but keeps stocks within a month together.
                                  # pin_memory=True When actually using CUDA.
                                  )
        
        # Prepare Validation Data for Early Stopping Checks
        X_val, y_val, mask_val, ids_val = prepare_tensor_data(
            df_val, feat_cols, target_col, MAX_UNIVERSE_SIZE, device
        )
        
        # --- 1. Early Stopping Setup ---
        MAX_POSSIBLE_EPOCHS = 100  # High ceiling, strictly controlled by patience
        PATIENCE = 20              # Stop if validation error increases/stagnates for 50 epochs
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch_found = 0
        
        # --- 2. Train with Early Stopping ---
        # Save the epoch if the validation error improved.   
        
        #Container to save model weights of best iteration
        best_model_state_temp = None
        
        # A. Training Step
        for epoch in range(MAX_POSSIBLE_EPOCHS):
            print(f"   Epoch: {epoch}/{MAX_POSSIBLE_EPOCHS}")
            epoch_loss = 0
            for b_X, b_y, b_mask, b_ids in train_loader: #b_ids contains the batch numbers to know which exact batches are being trained. 
                #    b_X shape is: [BATCH_SIZE, MAX_UNIVERSE_SIZE, feat_cols]
                #    The Transformer processes these BATCH_SIZE months in parallel.
                #    Gradients are averaged across the BATCH_SIZE months.
                #    optimizer.step() updates weights once per these 16 months.
                loss = train_step(temp_model, temp_optimizer, criterion, b_X, b_y, b_mask)
                epoch_loss += loss
                
            # B. Validation Step
            temp_model.eval()
            with torch.no_grad():
                preds_val = temp_model(X_val, src_key_padding_mask=mask_val)
                # Calculate loss only on valid (non-padded) data
                valid_preds = preds_val[~mask_val]
                valid_y = y_val[~mask_val]
                val_loss = criterion(valid_preds, valid_y).item()
                    
            # C. Early Stopping Logic 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch_found = epoch + 1 # Store as 1-based count
                # If desired: This will also save the model weights. best_model_state_temp = {k: v.detach().cpu().clone() for k, v in temp_model.state_dict().items()}
                patience_counter = 0 # Reset patience
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"      Early Stopping: Best epoch: {best_epoch_found}. Loss: {best_val_loss}")
                    break
        
        # --- 3. Save Found Hyperparameters ---
        saved_best_epochs = best_epoch_found
                
        # reset counter
        months_since_tune = 0  
    
    else:
        print(f"Skipping hyperparameter tuning at date {date}, reusing saved params.")
        
        if saved_best_epochs is None:
             raise ValueError("Error: Hyperparameters not found. Initial tuning step failed.")        
    
    # Increment months since last tune
    months_since_tune += 1

    # =======================================
    #    Refit on Train & Validation Data
    # =======================================
    
    # Get Train + Val Data   
    df_window = df[df['eom'].isin(window_months)]
    
    # Create Tensor Data
    X_window, y_window, mask_window, ids_window = prepare_tensor_data(
        df_window, feat_cols, target_col, MAX_UNIVERSE_SIZE, device
    )
    
    # Create DataLoader
    window_dataset = TensorDataset(X_window, y_window, mask_window)
    window_loader = DataLoader(window_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create Model
    final_model = StockTransformer(d_feat=len(feat_cols)).to(device)
    
    # Load the saved initialisation
    final_model.load_state_dict(saved_initial_state)
    
    final_optimizer = optim.AdamW(final_model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    final_criterion = nn.MSELoss()
    
    # Fit Model
    for epoch in range(saved_best_epochs):
        print(f"   Epoch: {epoch}/{saved_best_epochs}")
        for b_X, b_y, b_mask in window_loader:
             train_step(final_model, final_optimizer, final_criterion, b_X, b_y, b_mask) 
        
    
    # ==================================
    #   Predict next month's OOS Return
    # ===================================
    print(f"Predicting returns for date: {date}")
    # Identify the specific month we need to predict (the last available data point)
    # Logic: If 'date' is Feb 1st, we use data from Jan 31st to predict Feb returns
    pred_date_mask = (df['eom'] == date - pd.offsets.MonthEnd(1))
    df_test = df.loc[pred_date_mask]
    
    X_test, y_test, mask_test, ids_test = prepare_tensor_data(
        df_test, feat_cols, target_col, MAX_UNIVERSE_SIZE, device
    )
        
    # 2. Forward Pass (Inference)
    final_model.eval()
    with torch.no_grad():
        preds = final_model(X_test, src_key_padding_mask=mask_test)
        
        # 3. Extract Valid Predictions (Remove Padding)
        # ~mask_test selects the real data points (inverse of padding mask)
        valid_preds = preds[~mask_test].cpu().numpy()
        
        # 4. Align with IDs
        # ids_test_list matches the order of the batches. We concat them to match 'valid_preds'
        ids_test_df = pd.concat(ids_test, ignore_index=True)
        
        # 5. Sanity Check
        if len(valid_preds) != len(ids_test_df):
             print(f"Warning: Size mismatch. Preds {len(valid_preds)} vs IDs {len(ids_test_df)}")
    
    # ===================
    #  Save Predictions 
    # ===================
    #At 'eom', predict return for 'eom'+1
    pred_df = ids_test_df.copy()
    pred_df[prediction_name] = valid_preds
    predictions.append(pred_df)


    # =====================
    #  Save Model Weights
    # =====================
    filename = f"{model_name}_{target_type}_{file_end}_date_{date.strftime('%Y-%m-%d')}.pth"
    torch.save(final_model.state_dict(), os.path.join(model_save_path, filename))
    
#%% Save Predictions
df_predictions = pd.concat(predictions)

#At eom, the prediction is for eom+1
df_predictions.to_sql(name = f"{model_name}_{target_type}_{file_end}",
                   con = db_Predictions,
                   index = False,
                   if_exists = 'append')

JKP_Factors.close()
db_Predictions.close()
