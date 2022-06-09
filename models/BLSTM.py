from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import pandas as pd
import numpy as np
from tqdm import tqdm


# We reimplemented referring to the original Bayesian LSTM implementation in 
# Pytorch by Pawarit Laosunthara found at:
# https://github.com/PawaritL/BayesianLSTM
class BayesianLSTM(nn.Module):

    def __init__(self, input_size, output_length, batch_size, hidden_size_1=128, hidden_size_2=32, dropout_probability=0.5):
        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size # user-defined

        self.hidden_size_1 = hidden_size_1 # number of encoder cells (default value 128 from paper)
        self.hidden_size_2 = hidden_size_2 # number of decoder cells (default value 32 from paper)
        self.stacked_layers = 2 # number of (stacked) LSTM layers for each stage
        self.dropout_probability = dropout_probability # arbitrary value (the paper suggests that performance is generally stable across all ranges)

        self.lstm1 = nn.LSTM(input_size, 
                             self.hidden_size_1, 
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size_2, output_length)
        
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden(batch_size, self.hidden_size_1)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        
        state = self.init_hidden(batch_size, self.hidden_size_2)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        
        output = output[:, -1, :] # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred
        
    def init_hidden(self, batch_size, hidden_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, hidden_size))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, hidden_size))
        return hidden_state, cell_state
    
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()
    
    
def train_blstm(X_train, y_train, X_test, n_features, output_length, batch_size, n_epochs, learning_rate, sequence_length, n_experiments, target_scaler, split, save_model=True, load_model=None):
    
    
    if load_model == None: 
        bayesian_lstm = BayesianLSTM(n_features, output_length, batch_size)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=learning_rate)

        bayesian_lstm.train()

        loss_l = []
        times = []
        epochs = tqdm(range(1, n_epochs+1))
        for e in epochs:
            epochs.set_description('Training model ... Epochs')
            start_epoch = time.time()

            for b in range(0, len(X_train), batch_size):
                torch_features = X_train[b:b+batch_size,:,:]
                torch_target =  y_train[b:b+batch_size]    

                X_batch = torch.tensor(torch_features,dtype=torch.float32)    
                y_batch = torch.tensor(torch_target,dtype=torch.float32)

                output = bayesian_lstm(X_batch)
                loss = criterion(output.view(-1), y_batch)  

                loss.backward()
                optimizer.step()        
                optimizer.zero_grad() 
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            times.append(elapsed)
            
            epochs.set_postfix(loss=loss.item())
            loss_l += [loss.item()] 

        avg_time = sum(times)/n_epochs
        print('Total time elapsed:', sum(times))
        print('Average time per epoch:', avg_time)
        
        if save_model:
            torch.save(bayesian_lstm.state_dict(),
                       'trained_models/BLSTM'
                       +'_'+split
                       +'_seq'+str(sequence_length)
                       +'_feat'+str(n_features)
                       +'_bs'+str(batch_size)
                       +'_eps'+str(n_epochs)
                       +'_lr'+str(learning_rate)
                       +'.pth')
            
        pd.DataFrame(loss_l, columns=['Loss']).to_csv('trained_models/blstm_loss.csv')

    else:
        print('Loading model ...')
        bayesian_lstm = BayesianLSTM(n_features, output_length, batch_size)
        bayesian_lstm.load_state_dict(torch.load('trained_models/'+load_model))
        bayesian_lstm.eval()
    
    ### Monte Carlo Dropout
    experiment_predictions = []
    experiments = tqdm(range(n_experiments))
    for i in experiments:
        experiments.set_description('Monte Carlo Dropout ... Experiments')
        if target_scaler == None:
            experiment_predictions.append(bayesian_lstm.predict(X_test))
        else:
            experiment_predictions.append(target_scaler.inverse_transform(bayesian_lstm.predict(X_test)))
     
    y_pred = pd.DataFrame(np.array(experiment_predictions).T, 
                          columns=['R0_'+str(i) for i in range(n_experiments)])

    return(y_pred)


def calc_quantiles(y_pred, quantiles, labels):
    
    y_pred_quantiles = []
    for quantile, label in zip(quantiles, labels):
        print('Calculating '+label+' quantile:', quantile)
        y_pred_quantiles.append(y_pred.quantile(quantile, axis=1).values)
        
    y_pred_quantiles = pd.DataFrame(np.array(y_pred_quantiles).T, columns=labels)
    
    return(y_pred_quantiles)