import torch
import torch.backends
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset, TensorDataset
from data_prepare.feature_extract import audio_feature_extract, motion_feature_extract, load_skeleton
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from network import lstm
import numpy as np
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(0)
# is_train = True
# continue_train = False
continue_train = True
is_train = False


if continue_train or not is_train:
    last_epoch = 9140
model_name = 'LSTM-AE'
threshold = 0.06
# model_name = 'LSTM'
# Feature processing
with_tempo_normalized = True
with_tempo_features = True  # with tempo features
with_rotate = True
with_centering = False
with_leaky_relu = True
with_ortho_init = True
with_masking = True
one_sample_train = False

if model_name == 'LSTM-AE':
    with_tempo_features = True
    with_masking = True

"""
shared_parameters among Models
"""
acoustic_size = 50
temporal_size = 3
motion_size = 23 * 3
lr = 0.001
beta1 = 0.9
beta2 = 0.999
seq_len = 120
train_batch_size = 10
valid_batch_size = 1
test_batch_size = 1
num_workers = 2
max_epech = 10000
reduce_size = 10
"""
"""

"""
LSTM model parameters
"""
hidden_size = 80
fc1_size = 64
num_layers = 3  # 3 LSTM cells per time step



post_fix = model_name+("_rotate" if with_rotate else "")\
           +("_Ortho" if with_ortho_init else "")\
           +("_Leaky" if with_leaky_relu else "")\
           +("_Temporal" if with_tempo_features else "")\
           +("_OneSample" if one_sample_train else "")\
           +("_InputSize_%d" % acoustic_size)\
           +("_Seq_%d")%seq_len\
           +("_TempoNor" if with_tempo_normalized else "")
if model_name == 'LSTM-AE':
    post_fix = post_fix\
           +("_Threshold_%.3f"%threshold) \
           +("_Masking" if with_masking else "") \
           +("_Reduced_%d" % reduce_size )

# print(device)
data_dir = "../data/"
ck_dir= "../checkpoints/" + post_fix +"/"
output_json = post_fix+'.json'
if not os.path.exists(ck_dir):
    os.makedirs(ck_dir)

torch.backends.cudnn.benchmark = True

def load_features_from_dir(data_dir):
    acoustic_features,temporal_indexes = audio_feature_extract(data_dir)  # [n_samples, n_acoustic_features]
    motion_features = motion_feature_extract(data_dir, with_rotate=with_rotate, with_centering=with_centering)  # [n_samples, n_motion_features]
    return acoustic_features,temporal_indexes, motion_features[:acoustic_features.shape[0],:]

def load_train_features_and_scaler(train_dirs, acoustic_size, temporal_size, output_size, acoustic_scaler=MinMaxScaler(), motion_scaler=MinMaxScaler() ):
    print("\n...Feature extracting...\n")
    train_acoustic_features = np.empty([0,acoustic_size])
    train_temporal_features = np.empty([0,temporal_size])
    train_motion_features = np.empty([0,output_size])
    train_acoustic_scaler = acoustic_scaler
    train_motion_scaler = motion_scaler

    for one_dir in train_dirs:
        acoustic_features,temporal_indexes, motion_features = load_features_from_dir(one_dir)
        #  [n_samples, n_acoustic_features]

        train_acoustic_features = np.append(train_acoustic_features,acoustic_features,axis=0)
        train_motion_features = np.append(train_motion_features,motion_features,axis=0)
        train_temporal_features = np.append(train_temporal_features, temporal_indexes, axis=0)

    train_acoustic_features = train_acoustic_scaler.fit_transform(train_acoustic_features)
    train_motion_features = train_motion_scaler.fit_transform(train_motion_features)

    print("train size: %d" % (len(train_acoustic_features)))
    return train_acoustic_features, train_acoustic_scaler, train_motion_features, train_motion_scaler,train_temporal_features
    pass



def create_model(model_name):
    if model_name == 'LSTM':
        model = lstm.SimpleRNN(

            input_size=acoustic_size,
            fc1_size=fc1_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=motion_size,
            device=device,
            with_tempo=with_tempo_features,
            is_leaky_relu=with_leaky_relu).to(device)
        if with_ortho_init:
            orthogonal_init(model.rnn)

        pass

    elif model_name == 'LSTM-AE':
        model = lstm.LSTM_AE(
            input_size = acoustic_size,

            reduced_size=reduce_size,
            output_size=69,
            fc1_hidden_size= 24,
            fc2_hidden_size= 24,
            fc3_hidden_size= 24,
            encoder_rnn_hidden_size= 32,
            decoder_rnn_hidden_size= 32,
            pred_rnn_hidden_size= 32,
            num_layers= 3,
            with_masking=with_masking,
        ).to(device)
        if with_ortho_init:
            orthogonal_init(model.encoder_rnn)
            orthogonal_init(model.decoder_rnn)
            orthogonal_init(model.pred_rnn)
        pass
    else:
        return
        pass
    model = model.double()
    params = list(model.parameters())
    print(model)
    return model

def load_valid_features(valid_dirs):
    valid_acoustic_features = np.empty([0, acoustic_size])
    valid_temporal_features = np.empty([0,temporal_size])
    valid_motion_features = np.empty([0, motion_size])
    for one_dir in valid_dirs:
        acoustic_features, temporal_indexes, motion_features = load_features_from_dir(one_dir)
        # [n_samples, n_acoustic_features]

        valid_acoustic_features = np.append(valid_acoustic_features, acoustic_features, axis=0)
        valid_motion_features = np.append(valid_motion_features, motion_features, axis=0)
        valid_temporal_features = np.append(valid_temporal_features, temporal_indexes, axis=0)

    print("valid size: %d" % (len(valid_acoustic_features)))
    return valid_acoustic_features, valid_motion_features, valid_temporal_features
    pass

def orthogonal_init(m):
    assert isinstance(m,
                      (
                          nn.GRU,
                          nn.LSTM,
                          nn.GRUCell,
                          nn.LSTMCell
                      )
                      )
    for name, param in m.named_parameters():
        if name.find("weight_ih") >= 0:
            nn.init.xavier_uniform_(param)
        elif name.find("weight_hh") >= 0:
            nn.init.orthogonal_(param)
        elif name.find("bias") >= 0:
            nn.init.zeros_(param)
        else:
            raise NameError("unknown param {}".format(name))

def train(acoustic_features,val_acoustic_features, motion_features,val_motion_features, temporal_features, val_temporal_features):
    model = create_model(model_name=model_name)


    optimizer = optim.Adam(model.parameters(),lr = lr, betas = (beta1, beta2))
    epoch = 0

    if continue_train:
        checkpoint = torch.load(os.path.join(ck_dir, model_name + "_epoch_%d.pth"%last_epoch))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    assert (len(acoustic_features) == len(motion_features) == len(temporal_features))
    assert (len(val_acoustic_features) == len(val_motion_features) == len(val_temporal_features))
    if with_tempo_features:
        acoustic_features = np.hstack((acoustic_features, temporal_features))
        val_acoustic_features = np.hstack((val_acoustic_features, val_temporal_features))
    # Crop the input/target and drop the no-use data
    print(acoustic_features.shape)
    num_train_seq = int(len(acoustic_features) / seq_len)
    acoustic_features = acoustic_features[:num_train_seq*seq_len,:].reshape(num_train_seq,seq_len,-1)
    motion_features = motion_features[:num_train_seq*seq_len,:].reshape(num_train_seq,seq_len,-1)
    #
    num_val_seq = int(len(val_acoustic_features) / seq_len)
    val_acoustic_features = val_acoustic_features[:num_val_seq * seq_len, :].reshape(num_val_seq, seq_len, -1)
    val_motion_features = val_motion_features[:num_val_seq * seq_len, :].reshape(num_val_seq, seq_len, -1)
    # print(acoustic_features.shape)
    # x_train, x_valid, y_train, y_valid = train_test_split(acoustic_features, motion_features, shuffle=False)
    x_train, x_valid, y_train, y_valid = acoustic_features, val_acoustic_features, motion_features, val_motion_features
    x_train_data = torch.from_numpy(x_train)
    y_train_data = torch.from_numpy(y_train)
    x_valid_data = torch.from_numpy(x_valid)
    y_valid_data = torch.from_numpy(y_valid)

    loss_Ae = nn.MSELoss()
    loss_pred = nn.MSELoss()

    train_dataset = TensorDataset(x_train_data, y_train_data)
    valid_dataset = TensorDataset(x_valid_data, y_valid_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=num_workers)

    print("train size: %d, valid size: %d" % (train_dataloader.__len__() * train_batch_size, valid_dataloader.__len__() * valid_batch_size))

    min_valid_loss = float('inf')
    for epoch in range(epoch+1, epoch+max_epech):
        epoch_train_loss = 0
        epoch_train_auto_loss = 0
        epoch_train_pred_loss = 0
        epoch_valid_loss = 0
        model.train()
        # print(epoch)
        for i, (batch_x, batch_y) in enumerate(train_dataloader):
            # print("{0},{1}".format(batch_x.shape,batch_y.shape))
            batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)
            batch_acoustic_features = batch_x[:,:, :acoustic_size]
            optimizer.zero_grad()
            output = model(batch_x)
            if model_name == 'LSTM-AE':
                loss1 = torch.max(
                    torch.DoubleTensor([threshold]).to(device),
                    loss_Ae(output[0], batch_acoustic_features)
                )
                loss2 = loss_pred(output[1],batch_y)
                loss = loss1 + loss2
            else:
                loss = loss_pred(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            if model_name == 'LSTM-AE':
                epoch_train_auto_loss += loss1.item()
                epoch_train_pred_loss += loss2.item()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(valid_dataloader):
                batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)
                batch_acoustic_features = batch_x[:, :, :acoustic_size]
                output = model(batch_x)
                if model_name == 'LSTM-AE':
                    loss4 = torch.max(
                        torch.DoubleTensor([threshold]).to(device),
                        loss_Ae(output[0], batch_acoustic_features)
                    )
                    loss5 = loss_pred(output[1], batch_y)
                    loss3 = loss4 + loss5
                else:
                    loss3 = loss_pred(output, batch_y)
                epoch_valid_loss += loss3.item()


        print("epoch %d,train_AE_loss:%0.4f, train_Pred_loss:%0.4f, train_loss:%0.4f, valid_loss:%0.4f"
              % (epoch,  epoch_train_auto_loss/len(train_dataloader),epoch_train_pred_loss / len(train_dataloader),
                 epoch_train_loss/len(train_dataloader), epoch_valid_loss/len(valid_dataloader) ))
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}

        torch.save(state,f = os.path.join(ck_dir, model_name + "_epoch_%d.pth" % epoch))
        if epoch_valid_loss < min_valid_loss:
            min_valid_loss = epoch_valid_loss
            print("save latest min")
            # print("weight", model.fc2.weight.data.cpu().numpy(), "grad", model.fc2.weight.grad.data.cpu().numpy())
            # print("bias", model.fc2.bias.data.cpu().numpy(), "grad", model.fc2.bias.grad.data.cpu().numpy())
            torch.save(state,f = os.path.join(ck_dir, model_name + "_latest.pth"))

    pass



def test(test_sample_dir, acoustic_features_scaler, motion_features_scaler, tempo_scaler=None):
    print("\n........testing........\n")
    test_acoustic_features, temporal_indexes, test_motion_features = load_features_from_dir(test_sample_dir)
    if tempo_scaler is not None:
        temporal_indexes = tempo_scaler.transform(temporal_indexes)
    test_acoustic_features = acoustic_features_scaler.transform(test_acoustic_features)
    test_motion_features = motion_features_scaler.transform(test_motion_features)

    if with_tempo_features:
        test_acoustic_features = np.hstack((test_acoustic_features, temporal_indexes))

    # num_test_seq = int(len(test_acoustic_features) / seq_len)
    # test_acoustic_features = test_acoustic_features[:num_test_seq * seq_len, :].reshape(num_test_seq, seq_len, -1)
    # test_motion_features = test_motion_features[:num_test_seq * seq_len, :].reshape(num_test_seq, seq_len, -1)

    print("shape:{0}".format(test_acoustic_features.shape))
    model_path = os.path.join(ck_dir, model_name + "_epoch_%d.pth"%last_epoch)
    print("model load from %s" % model_path)
    checkpoint = torch.load(model_path)
    model = create_model(model_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    # test_dataset = TensorDataset(torch.from_numpy(test_acoustic_features[np.newaxis,:]), torch.from_numpy(test_motion_features[np.newaxis,:]))
    test_dataset = TensorDataset(torch.from_numpy(test_acoustic_features[np.newaxis,]), torch.from_numpy(test_motion_features[np.newaxis,]))
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=num_workers)

    criterion = nn.MSELoss()
    predict_motion_features = np.empty([0, motion_size])
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_dataloader):

            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            if model_name == 'LSTM-AE':
                loss = criterion(output[1], batch_y)
                output = np.reshape(output[1].detach().cpu().numpy(), newshape=[-1, motion_size])
                batch_y = np.reshape(batch_y.detach().cpu().numpy(), newshape=[-1, motion_size])

            else:
                # loss = criterion(output, batch_y)
                output = np.reshape(output.detach().cpu().numpy(), newshape=[-1, motion_size])
                batch_y = np.reshape(batch_y.detach().cpu().numpy(), newshape=[-1, motion_size])

                # output = output.detach().cpu().numpy()
            # loss_data = loss.detach().cpu().numpy()


            predict_motion_features = np.append(predict_motion_features, output,axis=0)

    predict_real_motion_features = motion_features_scaler.inverse_transform(predict_motion_features)
    predict_real_motion_features = np.reshape(predict_real_motion_features, newshape=[-1, motion_size // 3, 3])


    center = load_skeleton(os.path.join(test_sample_dir, 'skeletons.json'))[1][:len(predict_real_motion_features)]

    if with_centering:
        for i in range(len(predict_real_motion_features)):
            for j in range(len(predict_real_motion_features[i])):
                predict_real_motion_features[i][j] += center[i]

    data = dict()
    output_json_fn = os.path.join(test_sample_dir, output_json)
    data['length'] = len(predict_real_motion_features)
    data['center'] = center
    data['skeletons'] = predict_real_motion_features.tolist()
    with open(output_json_fn, 'w') as f:
        json.dump(data,f)
        print("saved as %s" % output_json_fn)
    pass

if __name__ == '__main__':
    train_dirs = [
        "../data/DANCE_C_1",
    ] if one_sample_train else[
        "../data/DANCE_C_1",
        "../data/DANCE_C_2",
        "../data/DANCE_C_3",
        "../data/DANCE_C_4",
        "../data/DANCE_C_6",
    ]
    valid_dirs = [
        "../data/DANCE_C_7",
        "../data/DANCE_C_8",
        "../data/DANCE_C_9",
    ]
    test_dirs = [
        "../data/DANCE_C_1",
        "../data/DANCE_C_2",
        "../data/DANCE_C_3",
        "../data/DANCE_C_4",
        "../data/DANCE_C_6",
        "../data/DANCE_C_7",
        "../data/DANCE_C_8",
        "../data/DANCE_C_9",
    ]


    acoustic_features, acoustic_features_scaler, motion_features,motion_features_scaler, temporal_features = \
        load_train_features_and_scaler(train_dirs=train_dirs,
                                       acoustic_size=acoustic_size,temporal_size=temporal_size,output_size=motion_size,
                                       acoustic_scaler=MinMaxScaler(), motion_scaler=MinMaxScaler())

    if with_tempo_normalized:
        temporal_features_scalar = MinMaxScaler()
        temporal_features = temporal_features_scalar.fit_transform(temporal_features)

    if is_train:
        val_acoustic_features, val_motion_features, val_temporal_features = load_valid_features(valid_dirs)
        val_acoustic_features = acoustic_features_scaler.transform(val_acoustic_features)
        if with_tempo_normalized:
            val_temporal_features = temporal_features_scalar.transform(val_temporal_features)
        val_motion_features = motion_features_scaler.transform(val_motion_features)
        train(acoustic_features,val_acoustic_features, motion_features,val_motion_features, temporal_features, val_temporal_features)
    else:
        if not with_tempo_normalized:
            test(test_dirs[6], acoustic_features_scaler, motion_features_scaler)
        else:
            test(test_dirs[6], acoustic_features_scaler, motion_features_scaler,temporal_features_scalar)


    pass
