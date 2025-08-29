import numpy as np
import pandas as pd
import tensorflow as tf
import random
import joblib
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Add,Permute,GlobalAveragePooling1D,Input,GlobalMaxPooling1D,SeparableConv1D, Lambda, Dense, Conv1D, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from tensorflow.keras.layers import Input, Dense, Lambda, LayerNormalization, MultiHeadAttention, Conv1D, BatchNormalization, GlobalAveragePooling1D

## custom loss for metric
class CustomLoss(Loss):
    def __init__(self, k_matrix):
        super().__init__()
        self.k_matrix = k_matrix

    def call(self, y_true, y_pred):
        diff = y_true - y_pred
        loss = diff * tf.convert_to_tensor(self.k_matrix,dtype=tf.float32) * diff
        return keras.backend.mean(loss)
        
## Model Architecture
def build_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    col_indices = tf.range(1, 301, dtype=tf.float32)  
    inputs =  col_indices + inputs  
    x = Lambda(lambda t: tf.expand_dims(t, axis=1))(inputs) #For Conv1D
    x = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(x) #Need 3d input
    mha = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    mha = LayerNormalization()(mha + x)  
    mha = GlobalAveragePooling1D()(mha)
    ffn = Dense(512, activation="relu")(mha)
    ffn = Dense(64, activation="relu")(ffn)
    ffn = Dense(output_dim, activation="linear")(ffn)
    model = keras.Model(inputs, ffn)  
    print(model.summary())
    return model

def cross_validate(X, y, X_test, k_matrix, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    val_losses = []
    y_oof = np.zeros_like(y, dtype=np.float32)
    y_test=[]
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training on fold {fold + 1}/{n_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        #scale to x
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test1 = scaler.transform(X_test)

        #scale to y
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)

        model = build_model(input_dim=X.shape[1], output_dim=y.shape[1])
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss=CustomLoss(k_matrix))
        history = model.fit(X_train, 
                            y_train,
                            validation_data=(X_val, y_val),
                            epochs=20,
                            batch_size=128,
                            verbose=2)

        val_loss = history.history['val_loss'][-1]
        val_losses.append(val_loss)
        print(f"Fold {fold + 1} validation loss: {val_loss}")
        y_oof[val_idx] = np.squeeze(scaler_y.inverse_transform(model.predict(X_val,verbose=0)))
        y_test.append(np.squeeze(scaler_y.inverse_transform(model.predict(X_test1,verbose=0))))
    avg_val_loss = np.mean(val_losses)
    print(f"Average validation loss across {n_folds} folds: {avg_val_loss}")
    return avg_val_loss,y_oof,y_test


def load_preprocessed_data(filename, n_to_sample, num_lines, n_folds):
    if num_lines <= 0:
        print("Empty.")
    else:
        if n_to_sample > num_lines:
            print(f" ({n_to_sample}) > ({num_lines}).Will read all rows.")
            df = pd.read_csv(filename)
        else:
            raise NotImplementedError("Sampling is not consistent.")
            all_indices = list(range(num_lines))
            sample_indices = sorted(random.sample(all_indices, n_to_sample))
            skip_rows = sorted(random.sample(range(1, num_lines + 1), num_lines - n_to_sample))
            rows_to_keep_0_indexed = sorted(random.sample(range(num_lines), n_to_sample))
            merge_df = pd.read_csv(filename,
                            skiprows=lambda i: i > 0 and i -1 not in rows_to_keep_0_indexed).fillna(0)
            
    exit(0)


    X = np.array(train.iloc[:,0:300+VECTOR_SIZE+NEW_FEATURE]).astype('float32')
    y = np.array(train.iloc[:,300+VECTOR_SIZE+NEW_FEATURE:600+VECTOR_SIZE+NEW_FEATURE]).astype('float32')
    X_test = np.array(test.iloc[:,0:300+VECTOR_SIZE+NEW_FEATURE]).astype('float32')


    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    val_losses = []
    y_oof = np.zeros_like(y, dtype=np.float32)
    y_test=[]
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training on fold {fold + 1}/{n_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        #scale to x
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test1 = scaler.transform(X_test)

        #scale to y
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)

        model = build_model(input_dim=X.shape[1], output_dim=y.shape[1])
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss=CustomLoss(k_matrix))
        history = model.fit(X_train, 
                            y_train,
                            validation_data=(X_val, y_val),
                            epochs=20,
                            batch_size=128,
                            verbose=2)

        val_loss = history.history['val_loss'][-1]
        val_losses.append(val_loss)
        print(f"Fold {fold + 1} validation loss: {val_loss}")
        y_oof[val_idx] = np.squeeze(scaler_y.inverse_transform(model.predict(X_val,verbose=0)))
        y_test.append(np.squeeze(scaler_y.inverse_transform(model.predict(X_test1,verbose=0))))
    avg_val_loss = np.mean(val_losses)
    print(f"Average validation loss across {n_folds} folds: {avg_val_loss}")
    return avg_val_loss,y_oof,y_test


class PredictorGold:
    def __init__(self, root_dir='trained_models_local', n_folds = 5):
        self.n_folds = n_folds
        self.models = self.load_models(root_dir, n_folds)
        self.scalers_X, self.scalers_y = self.load_scalers(root_dir)

    def predict(self, X):
        y_test = []
        for i in range(self.n_folds):
            X_test1 = self.scalers_X[i].transform(X)
            y_test.append(np.squeeze(self.scalers_y[i].inverse_transform(
                self.models[i].predict(X_test1,verbose=0))))
        result = np.mean(y_test,axis=0)
        return result

    def load_scalers(self, root_dir='trained_models_local', n_folds=5):
        scalers_X = []
        scalers_y = []
        for i in range(n_folds):
            scaler_X = joblib.load(f"{root_dir}/scaler_X_fold_{i}.pkl")
            scaler_y = joblib.load(f"{root_dir}/scaler_y_fold_{i}.pkl")
            scalers_X.append(scaler_X)
            scalers_y.append(scaler_y)
        return scalers_X, scalers_y

    def load_models(self, root_dir='trained_models_local', n_folds=5):
        models = [] 
        for i in range(n_folds):
            model = build_model(input_dim=300, output_dim=300)
            model.load_weights(f"trained_models/model_fold_{i}.h5")  # Load weights for each fold
            models.append(model)

        return models
    

if __name__ == "__main__":
    test = pd.read_csv("kaggle_data/test.csv").fillna(0)
    test = test.drop('geology_id', axis=1)

    predictor_gold = PredictorGold(root_dir='trained_models_local', n_folds=5)

    results = predictor_gold.predict(test.values)

    print(results)
