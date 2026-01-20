import speck as sp
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable = True) 


pairs = 2
nr = 6
# wdir = './teacher_distinguisher/'
# # net= load_model(wdir+'speck_model_7r_num_epochs20_acc_0.8128569722175598.h5')
# net= load_model(wdir+'speck_model_8r_num_epochs10_acc_0.5532829761505127.h5')

wdir = './teacher_distinguisher/'
# net= load_model(wdir+'speck_model_7r_num_epochs40_acc_0.805361.h5')
net= load_model(wdir+'speck_model_6r_num_epochs20_acc_0.8667299747467041.h5')

def evaluate(net,X,Y):

    batch_size = 2000 
    Z = net.predict(X,batch_size=batch_size).flatten();
    print("teacher acc:", np.mean((Z > 0.5) == Y))
    print("Z mean/std:", Z.mean(), Z.std())
    print("pos mean:", Z[Y == 1].mean(), "neg mean:", Z[Y == 0].mean())
    print("frac in [0.45,0.55]:", np.mean((Z > 0.45) & (Z < 0.55)))
    Zbin = (Z >= 0.5);
    diff = Y - Z;
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;

    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr);


X,Y = sp.make_train_data(10**6,nr,pairs);


evaluate(net, X, Y);


