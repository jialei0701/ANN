from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from layers import  GraphConvolution

from utils import *

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GCN(object):
    def __init__(self, dataset='cora', epochs=200, early_stopping=10):
        # Define parameters
        self.dataset = dataset

        self.NB_EPOCH = epochs
        self.PATIENCE = early_stopping

        # Get data
        self.X, self.A, self.y = load_data(dataset=self.dataset)
        self.y_train, self.y_val, self.y_test, self.idx_train, self.idx_val, self.idx_test, self.train_mask = get_splits(self.y)

        # Normalize X
        self.X /= self.X.sum(1).reshape(-1, 1)

        # pre-process A
        self.A_ = preprocess_adj(self.A)
        self.support = 1
        self.graph = [self.X, self.A_]

        self.pred_labels = None

    def train(self,hidden_num=16, dropout_rate=0.5, l2_reg=5e-4, learning_rate=0.01):
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
        X_in = Input(shape=(self.X.shape[1],))
        H = Dropout(dropout_rate)(X_in)
        H = GraphConvolution(hidden_num, self.support, activation='relu', kernel_regularizer=l2(l2_reg))([H]+G)
        H = Dropout(dropout_rate)(H)
        Y = GraphConvolution(self.y.shape[1], self.support, activation='softmax')([H]+G)

        # Compile model
        model = Model(inputs=[X_in]+G, outputs=Y)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))

        # Helper variables for main training loop
        wait = 0
        best_val_loss = 99999

        # Fit
        for epoch in range(1, self.NB_EPOCH+1):

            # Log wall-clock time
            t = time.time()

            # Single training iteration (we mask nodes without labels for loss calculation)
            model.fit(self.graph, self.y_train, sample_weight=self.train_mask,
                      batch_size=self.A.shape[0], epochs=1, shuffle=False, verbose=0)

            # Predict on full dataset
            self.pred_labels = model.predict(self.graph, batch_size=self.A.shape[0])

            # Train / validation scores
            train_val_loss, train_val_acc = evaluate_preds(self.pred_labels, [self.y_train, self.y_val],
                                                           [self.idx_train, self.idx_val])
            print("Epoch: {:04d}".format(epoch),
                  "train_loss= {:.4f}".format(train_val_loss[0]),
                  "train_acc= {:.4f}".format(train_val_acc[0]),
                  "val_loss= {:.4f}".format(train_val_loss[1]),
                  "val_acc= {:.4f}".format(train_val_acc[1]),
                  "time= {:.4f}".format(time.time() - t))

            # Early stopping
            if train_val_loss[1] < best_val_loss:
                best_val_loss = train_val_loss[1]
                wait = 0
            else:
                if wait >= self.PATIENCE:
                    print('Epoch {}: early stopping'.format(epoch))
                    break
                wait += 1

    def test(self):
        # Testing
        test_loss, test_acc = evaluate_preds(self.pred_labels, [self.y_test], [self.idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(test_loss[0]),
              "accuracy= {:.4f}".format(test_acc[0]))

gcn = GCN()
gcn.train()
gcn.test()
'''
result:
Epoch: 0195 train_loss= 0.4884 train_acc= 0.9643 val_loss= 0.8339 val_acc= 0.8267 time= 0.1143
Epoch: 0196 train_loss= 0.4862 train_acc= 0.9643 val_loss= 0.8318 val_acc= 0.8267 time= 0.1107
Epoch: 0197 train_loss= 0.4840 train_acc= 0.9643 val_loss= 0.8303 val_acc= 0.8300 time= 0.1057
Epoch: 0198 train_loss= 0.4814 train_acc= 0.9643 val_loss= 0.8285 val_acc= 0.8300 time= 0.1057
Epoch: 0199 train_loss= 0.4789 train_acc= 0.9643 val_loss= 0.8269 val_acc= 0.8267 time= 0.1137
Epoch: 0200 train_loss= 0.4764 train_acc= 0.9643 val_loss= 0.8253 val_acc= 0.8300 time= 0.1088
Test set results: loss= 0.8872 accuracy= 0.8090
'''