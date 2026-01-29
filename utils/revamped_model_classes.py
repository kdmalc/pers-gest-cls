import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import deque

from utils.gesture_dataset_classes import *


class SmoothedEarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, smoothing_window=5):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.best_loss = np.inf
        self.num_bad_epochs = 0
        self.loss_buffer = deque(maxlen=smoothing_window)

    def __call__(self, current_loss):
        self.loss_buffer.append(current_loss)
        smoothed_loss = np.mean(self.loss_buffer)

        if smoothed_loss < self.best_loss - self.min_delta:
            self.best_loss = smoothed_loss
            self.num_bad_epochs = 0  # Reset patience if improvement
        else:
            self.num_bad_epochs += 1  # Increment if no improvement

        return self.num_bad_epochs >= self.patience  # Stop if patience exceeded


class EarlyStopping:
    # This has been mostly depreciated (still used in AMC clustering code...)
    ## This version actually has everything except the smoothing... 
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
    

def select_model(model_type, config):
    if isinstance(model_type, str):
        if model_type in ['CNN', 'DynamicCNN', 'OriginalELEC573CNN']:  # ?D
            model = DynamicCNN(config)
        #elif model_type in ['CNNLSTM', 'DynamicCNNLSTM']:  # ?D
        #    model = DynamicCNNLSTM(config)
        elif model_type in ['CTRLNet']:  # ?D
            model = CTRLNet(config)
        #elif model_type == 'ELEC573Net':  # 2D:
        #    model = ELEC573Net(config)
        elif model_type == "DynamicMomonaNet":  # 2D
            model = DynamicMomonaNet(config)
        elif model_type == "CNNModel3layer":
            model = CNNModel_3layer(config)
        else:
            raise ValueError(f"{model_type} not recognized.")
    else:
        model = model_type

    return model


class CTRLNet(nn.Module):
    # Meta CTRL Labs Generic Neuromotor Interface
    def __init__(self, config):
        super(CTRLNet, self).__init__()

        self.config = config
        self.input_channels = config["num_channels"]
        self.num_classes = config["num_classes"]
        
        conv_hypparams = config["conv_layers"][0]  # ONLY SUPPORTS 1 LAYER CNNS!!!
        self.hidden_size = conv_hypparams[0]  # This is how many conv filters to use, and also how big the LSTM is
        self.kernel_size = conv_hypparams[1]
        self.stride = conv_hypparams[2]

        self.num_lstm_layers = config["lstm_num_layers"]
        
        self.conv = nn.Conv1d(in_channels=self.input_channels, out_channels=self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=config["padding"])
        self.ln1 = nn.LayerNorm(self.hidden_size)
        
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers, batch_first=True)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(config["fc_dropout"])

    def forward(self, x):
        #print(f"Forward start: {x.shape}")
        # x: [batch_size, channels, seq_len]
        #x = x.reshape(-1, self.config["num_channels"], self.config["sequence_length"])
        x = self.conv(x)  # [B, H, T']
        #print(f"Post conv: {x.shape}")
        x = x.permute(0, 2, 1)  # [B, T', H]
        #print(f"Post permute: {x.shape}")
        x = self.ln1(x)
        #print(f"Post ln1: {x.shape}")
        
        x, _ = self.lstm(x)
        #print(f"Post lstm: {x.shape}")
        x = self.ln2(x)
        #print(f"Post ln2: {x.shape}")
        
        # I had dropout applied after the FC (ie on the logits), which I think was actually wrong
        # Double check their paper to make sure dropout is now applied in the right place
        x = self.dropout(x)
        out = self.fc(x[:, -1, :])  # Use last timestep
        #print(f"Out: {out.shape}")
        return out
    
    def forward_features(self, x):
        # x: [batch_size, channels, seq_len]
        x = self.conv(x)  # [B, H, T']
        x = x.permute(0, 2, 1)  # [B, T', H]
        x = self.ln1(x)
        x, _ = self.lstm(x)
        x = self.ln2(x)
        
        # The latent representation before the FC layer, using last timestep:
        feats = x[:, -1, :]
        return feats


class DynamicMomonaNet(nn.Module):
    """CNN-LSTM model that can be changed according to its config, mainly for optuna"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nc = config['num_channels']
        self.sl = config['sequence_length']
        self.cnn_dropout = config['cnn_dropout']
        self.lstm_dropout = config['lstm_dropout']
        self.fc_dropout = config['fc_dropout']
        self.use_dense_cnn_lstm = config["use_dense_cnn_lstm"]
        self.lstm_num_layers = config["lstm_num_layers"]

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # --- Convolutional layers (with dropout) ---
        self.conv_layers = nn.ModuleList()
        in_channels = self.nc
        for conv_layer in config['conv_layers']:
            out_channels, kernel_size, stride = conv_layer
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=config["padding"]),
                nn.BatchNorm1d(out_channels) if config.get("use_batch_norm", False) else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.cnn_dropout) if self.cnn_dropout > 0 else nn.Identity()
            ))
            in_channels = out_channels

        # --- Per-layer pooling setup ---
        self.pooling_layers = config.get('pooling_layers', [True] * len(config['conv_layers']))
        self.pool = nn.MaxPool1d(config["maxpool_kernel_size"]) if config["use_layerwise_maxpool"] else nn.Identity()

        # --- Global pooling (always ON) ---
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Always ON

        # --- Dense layer between CNN and LSTM ---
        if self.use_dense_cnn_lstm:
            self.dense_cnn_lstm = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.Dropout(self.lstm_dropout) if self.lstm_dropout > 0 else nn.Identity()
            )

        # --- Conditional LSTM ---
        if self.lstm_num_layers > 0:
            self.use_lstm = True
            self.lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=config['lstm_hidden_size'],
                num_layers=self.lstm_num_layers,
                batch_first=True,
                dropout=config['lstm_dropout'] if self.lstm_num_layers > 1 else 0
            )
        else:
            self.use_lstm = False

        # --- Dummy input for FC in_features calculation ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.nc, self.sl)
            x = dummy_input
            for i, conv in enumerate(self.conv_layers):
                x = conv(x)
                if self.pooling_layers[i]:
                    x = self.pool(x)
            x = self.global_pool(x)
            x = x.flatten(1)
            if self.use_lstm:
                x = x.unsqueeze(1)  # shape: (batch, seq=1, channels)
                if self.use_dense_cnn_lstm:
                    x = self.dense_cnn_lstm(x)
                x, _ = self.lstm(x)
                x = x.flatten(1)
            in_features = x.shape[1]

        # --- Fully connected layers ---
        self.fc_layers = nn.ModuleList()
        for out_features in config['fc_layers']:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Dropout(self.fc_dropout) if self.fc_dropout > 0 else nn.Identity()
            ))
            in_features = out_features

        # --- Output layer ---
        self.output_layer = nn.Linear(in_features, config['num_classes'])

    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.pooling_layers[i]:
                x = self.pool(x)
        # TODO: Global pooling BEFORE LSTM is not best practice since it sets sl=1...
        ## LSTM with Sequence Length 1: Processes only one input, so "memory" and sequence logic are never utilized.
        x = self.global_pool(x)
        x = x.flatten(1)
        if self.use_lstm:
            x = x.unsqueeze(1)  # (batch, seq=1, channels)
            if self.use_dense_cnn_lstm:
                x = self.dense_cnn_lstm(x)
            x, _ = self.lstm(x)
            x = x.flatten(1)
        for fc in self.fc_layers:
            x = fc(x)
        logits = self.output_layer(x)
        return logits
    
    def forward_features(self, x):
        """
        Returns the features right before the final FC (output) layer.
        Passes through all layers except the last output FC.
        """
        # CNN + (per-layer pooling)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.pooling_layers[i]:
                x = self.pool(x)

        # Do NOT globally pool before LSTM (see best practices, below)
        # Instead, keep as [B, C, T] for LSTM

        # Prepare input for LSTM: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)
        if self.use_dense_cnn_lstm:
            x = self.dense_cnn_lstm(x)

        if self.use_lstm:
            x, _ = self.lstm(x)
            # x is [B, T, H], take the last time step
            x = x[:, -1, :]
        else:
            # If no LSTM, just flatten the features
            x = x.flatten(1)

        # Pass through all but the final FC layer
        for fc in self.fc_layers[:-1]:
            x = fc(x)
        feats = x  # Pre-final-FC features

        return feats
    

class DynamicCNN(nn.Module):
    def __init__(self, config):
        super(DynamicCNN, self).__init__()
        self.config = config
        self.input_channels = config["num_channels"]
        self.num_classes = config["num_classes"]
        self.use_batch_norm = config["use_batch_norm"]

        # Activation & Pooling
        self.relu = nn.ReLU()
        
        # Convolutional Layers
        self.conv_layers = nn.ModuleList()

        in_channels = self.input_channels
        for layer_spec in config["conv_layers"]:
            out_channels, kernel_size, stride = layer_spec
            conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=config["padding"]
            )
            bn = nn.BatchNorm1d(out_channels) if self.use_batch_norm else nn.Identity()
            
            # Sequential block with optional maxpool
            self.conv_layers.append(nn.Sequential(
                conv,
                bn,
                self.relu,
                nn.MaxPool1d(config["maxpool_kernel_size"]) if config["use_layerwise_maxpool"] else nn.Identity()
            ))
            in_channels = out_channels

        # Adaptive pooling handles variable lengths
        ## TODO: Really ought to tune this value... idk what the incoming shape is tho...
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Fully Connected Layers
        self.fc_layers = nn.ModuleList()
        fc_input_dim = in_channels  # Output channels from last conv layer
        
        for i, fc_size in enumerate(config["fc_layers"]):
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_size))
            if i < len(config["fc_layers"]) - 1:  # Apply ReLU to all but last layer
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(config["fc_dropout"]))  # Should the last FC layer have dropout? NO!
            fc_input_dim = fc_size

        # Final classification layer
        self.final_fc = nn.Linear(fc_input_dim, self.num_classes)

    def forward(self, x):
        #x = x.unsqueeze(1)  # Add channel dimension
        #x = x.view(x.shape[0], self.input_channels, -1)  # (batch, 16, seq_len)
        #print(f"x.shape (after initial reshape {x.shape}")
        
        # Apply conv layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Adaptive pooling ensures fixed size output
        ## Should this be always on or only on when config["use_layerwise_maxpool"] is False? Hardcoding that to False either way tho ig
        ## Global pool is fine with moments FE since it reduces from 5-->1. With actual time series data might want to test its effect...
        x = self.global_pool(x)  # Shape: (batch, channels, 1)
        x = x.flatten(1)  # Shape: (batch, channels)
        
        # Fully connected layers
        for fc_block in self.fc_layers:
            x = fc_block(x)
        
        return self.final_fc(x)
    

class CNNModel_3layer(nn.Module):
    # ORIGINAL ELEC573 NET RECREATION
    def __init__(self, config):
        super(CNNModel_3layer, self).__init__()
        self.input_dim = config['num_channels']
        #print(f"self.input_dim: {self.input_dim}")
        self.num_classes = config['num_classes']
        self.config = config

        # Activation and Pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(config["maxpool_kernel_size"])
        self.softmax = nn.Softmax(dim=1)
        
        # Convolutional Layers
        out_ch0, ks0, st0 = config["conv_layers"][0]
        self.conv1 = nn.Conv1d(self.input_dim, out_ch0,  # Should this be self.input_dim or just 1? Was 1...
                            kernel_size=ks0, 
                            stride=st0, 
                            padding=config["padding"])
        self.bn1 = nn.BatchNorm1d(out_ch0) if config["use_batch_norm"] else nn.Identity()
        
        out_ch1, ks1, st1 = config["conv_layers"][1]
        self.conv2 = nn.Conv1d(out_ch0, out_ch1, 
                            kernel_size=ks1, 
                            stride=st1, 
                            padding=config["padding"])
        self.bn2 = nn.BatchNorm1d(out_ch1) if config["use_batch_norm"] else nn.Identity()
        
        out_ch2, ks2, st2 = config["conv_layers"][2]
        self.conv3 = nn.Conv1d(out_ch1, out_ch2, 
                            kernel_size=ks2, 
                            stride=st2, 
                            padding=config["padding"])
        self.bn3 = nn.BatchNorm1d(out_ch2) if config["use_batch_norm"] else nn.Identity()
        
        # Dynamically calculate flattened size
        assert(self.input_dim!=1)
        #test_input = torch.randn(1, 1, self.input_dim)
        test_input = torch.randn(1, self.input_dim, config["sequence_length"])
        # Run through conv layers to calculate final size
        with torch.no_grad(): 
            #print(f"test_input.shape: {test_input.shape}")
            test_x = self.conv1(test_input)
            #print(f"Shape after conv1: {test_x.shape}")
            test_x = self.maxpool(test_x)
            #print(f"Shape after maxpool: {test_x.shape}")
            test_x = self.conv2(test_x)
            #print(f"Shape after conv2: {test_x.shape}")
            test_x = self.maxpool(test_x)
            #print(f"Shape after maxpool: {test_x.shape}")
            test_x = self.conv3(test_x)
            #print(f"Shape after conv3: {test_x.shape}")
            #if test_x.shape[-1]>1:
            #    test_x = self.maxpool(test_x)
            #    #print(f"Shape after maxpool: {test_x.shape}")
            flattened_size = test_x.view(1, -1).size(1)
            #print(f"Shape after flattening: {test_x.shape}")
            #print(f"flattened_size of test input: {flattened_size}")
        
        # Fully Connected Layers
        ## Assumes only 1 FC weights set
        self.fc1 = nn.Linear(flattened_size, config["fc_layers"][0])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc2 = nn.Linear(config["fc_layers"][0], self.num_classes)

    def forward(self, x):
        #print(f"Input x shape: {x.shape}")
        # Ensure input is the right shape
        ## Why was this unsqueeze necessary? 
        #x = x.unsqueeze(1)  # Reshape input to (batch_size, 1, sequence_length)
        #assert x.shape[1] == 1, f"Expected 1 input channel, got {x.shape[1]}"
        #print(f"After unsqueeze: {x.shape}")
        
        # Conv Block 1
        x = self.conv1(x)
        #print(f"After conv1: {x.shape}")
        x = self.bn1(x)
        #print(f"After bn1: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.maxpool(x)
        #print(f"After maxpool: {x.shape}")
        
        # Conv Block 2
        x = self.conv2(x)
        #print(f"After conv2: {x.shape}")
        x = self.bn2(x)
        #print(f"After bn2: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.maxpool(x)
        #print(f"After maxpool: {x.shape}")
        
        # Conv Block 3
        x = self.conv3(x)
        #print(f"After conv3: {x.shape}")
        x = self.bn3(x)
        #print(f"After bn3: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        #x = self.maxpool(x)
        #print(f"After maxpool: {x.shape}")
        
        # Flatten and Fully Connected Layers
        x = x.view(x.size(0), -1)  # Flatten while preserving batch size
        #print(f"After flattening for FC: {x.shape}")
        x = self.fc1(x)
        #print(f"After fc1: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.dropout(x)
        #print(f"After dropout: {x.shape}")
        x = self.fc2(x)
        #print(f"After fc2: {x.shape}")
        #x = self.softmax(x)
        #print(f"After softmax: {x.shape}")
        return x


