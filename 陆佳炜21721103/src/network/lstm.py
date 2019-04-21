import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import  Variable


# Device configuration

# Hyper-parameters



# Recurrent neural network (many-to-one)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, fc1_size, hidden_size, num_layers, output_size, device, with_tempo, is_leaky_relu):
        super(SimpleRNN, self).__init__()
        self.acoustic_features = input_size

        self.temporal_features = 3 if with_tempo else 0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.rnn = nn.LSTM(fc1_size , hidden_size, num_layers, batch_first=True, dropout=0.2)
        # self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_size+ self.temporal_features, fc1_size)
        self.relu1 = nn.LeakyReLU() if is_leaky_relu else nn.ReLU()
        self.fc = nn.Linear(hidden_size, 80)
        self.relu2 = nn.LeakyReLU() if is_leaky_relu else nn.ReLU()
        self.fc2 = nn.Linear(80, output_size)
        self.device = device

    def forward(self, x):
        x = self.relu1(self.fc1(x))

        # x1 = torch.cat((x,x_temp),dim=2)
        # Set initial hidden and cell states
        h0, c0 = self.init_hidden(x.size(0))

        # Forward propagate LSTM
        out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc2(self.relu2(self.fc(out)))
        return out

    def init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(self.num_layers,batch_size, self.hidden_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(self.num_layers,batch_size, self.hidden_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()


class LSTM_AE(nn.Module):
    def __init__(self, input_size,output_size,reduced_size,
                 fc1_hidden_size,fc2_hidden_size,fc3_hidden_size,
                 encoder_rnn_hidden_size,decoder_rnn_hidden_size,pred_rnn_hidden_size,
                 num_layers,with_masking ):
        super(LSTM_AE, self).__init__()
        self.acoustic_features = input_size
        self.with_masking = with_masking
        self.temporal_features = 3
        self.hidden_size = encoder_rnn_hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_size, fc1_hidden_size),
            nn.ReLU(),
        )
        self.encoder_rnn = nn.LSTM(fc1_hidden_size + self.temporal_features,encoder_rnn_hidden_size, 1, batch_first=True)
        self.encoder_fc2 = nn.Linear(encoder_rnn_hidden_size, reduced_size)


        self.decoder_fc = nn.Sequential(
            nn.Linear(reduced_size, fc2_hidden_size),
            nn.ReLU(),
        )
        self.decoder_rnn=nn.LSTM(fc2_hidden_size + self.temporal_features, decoder_rnn_hidden_size, 1, batch_first=True)
        self.decoder_fc2 = nn.Linear(decoder_rnn_hidden_size, input_size)

        self.pred_fc = nn.Sequential(
            nn.Linear(reduced_size, fc3_hidden_size),
            nn.ReLU(),
        )
        self.pred_rnn = nn.LSTM(fc3_hidden_size + self.temporal_features, pred_rnn_hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.pred_fc2 = nn.Linear(pred_rnn_hidden_size, output_size )


    def forward(self, x):
        x_ac = x[:,:,:self.acoustic_features]
        x_tempo = x[:,:, -self.temporal_features:]
        beat = x[:,:, -2]
        encoder_fc_out = self.encoder_fc(x_ac)
        encoder_rnn_in = torch.cat((encoder_fc_out, x_tempo), dim=2)
        # Set initial hidden and cell states
        h0, c0 = self.init_hidden(encoder_rnn_in.size(0),1)
        encoder_rnn_out,_ = self.encoder_rnn(encoder_rnn_in,(h0,c0))
        encoder_out = self.encoder_fc2(encoder_rnn_out)

        if(self.with_masking):
            mask = beat # beat frame
            for i in range(encoder_out.shape[-1]):
                encoder_out[:,:,i] = encoder_out[:,:,i].mul_(mask)


        decoder_fc_out = self.decoder_fc(encoder_out)
        decoder_rnn_in = torch.cat((decoder_fc_out, x_tempo), dim=2)
        h1, c1 = self.init_hidden(decoder_rnn_in.size(0),1)
        decoder_rnn_out,_ = self.decoder_rnn(decoder_rnn_in, (h1,c1))
        decoder_out = self.decoder_fc2(decoder_rnn_out)

        pred_fc_out = self.pred_fc(encoder_out)
        pred_rnn_in = torch.cat((pred_fc_out, x_tempo), dim=2)
        h2, c2 = self.init_hidden(pred_rnn_in.size(0),3)
        pred_rnn_out,_ = self.pred_rnn(pred_rnn_in, (h2,c2))
        pred_out = self.pred_fc2(pred_rnn_out)


        return [decoder_out,pred_out]

    def init_hidden(self, batch_size, num_layers):
        hidden = Variable(next(self.parameters()).data.new(num_layers,batch_size, self.hidden_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(num_layers,batch_size, self.hidden_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()
