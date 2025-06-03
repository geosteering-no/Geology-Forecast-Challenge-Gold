# Geology-Forecast-Challenge
Geology Forecast Challenge 1st solution

## How to run 
All the code runs on Kaggle. All the training runs on Kaggle P100. You can run it there as you like.

All the libraries used are in their default versions of kaggle. When you copy a notebook on Kaggle, the environment is copied too.

It takes about 20 minutes to run.


["Generate dataset"](https://www.kaggle.com/code/act18l/generate-dataset)

["Single Model"](https://www.kaggle.com/code/act18l/single-model-for-geology-forecast-challenge) 

The dataset ["geo_submission"](https://www.kaggle.com/datasets/act18l/geo-submission/data) contains  the results from  versions of the above notebooks.


["Emsemble"](https://www.kaggle.com/code/act18l/ensemble-for-geology-forecast-challenge)

In the competition, I chose to  **"Single Model(Version 25)"** and **"Ensemble(Version 9)"** as my submission.

You can use "Compare Versions" to check the code changes.

## Data Section
I used the original data and the provided interpolation script to extract more training data. Finally, I obtained 314,360 samples.

But due to memory constraints, Kaggle couldn't train on all samples. So I used as many samples as possible within a memory threshold of about 210,000. That is, I randomly selected 210,000 samples for each training session.

For feature engineering, I first manually added some simple features, but they didn't improve performance. So I abandoned traditional feature engineering and let the neural network extract features instead. 

For the "geology_id" column, I tried many methods like Word2Vec and KNN, but they reduced the model's performance. So, I dropped this column.

I used `StandardScaler` on the data. Since the input and target variables are essentially the same, I applied `StandardScaler` to both X and y, rather than just scaling the inputs as commonly done.


## Training and Model
I use `Kfold(5)`.

As mentioned [here](https://www.kaggle.com/competitions/geology-forecast-challenge-open/discussion/569884), the  metric is actually equivalent to weighted MSE. So we can directly use it as the optimization metric instead of the default MSE.
```python
class CustomLoss(Loss):
    def __init__(self, k_matrix):
        super().__init__()
        self.k_matrix = k_matrix

    def call(self, y_true, y_pred):
        diff = y_true - y_pred
        loss = diff * tf.convert_to_tensor(self.k_matrix,dtype=tf.float32) * diff
        return keras.backend.mean(loss)
```
For the model, it's based on my competition experience in the '[Forecasting Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1)' competition, where I referenced these two [notebooks](https://www.kaggle.com/code/act18l/convnet-starter-lb-0-052) and [notebooks](https://www.kaggle.com/code/cdeotte/transformer-starter-lb-0-052).

Aiming for efficiency, I focus on building faster and more accurate neural networks.

```python
def build_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    col_indices = tf.range(1, 301, dtype=tf.float32)  
    inputs =  col_indices + inputs  
    x = Lambda(lambda t: tf.expand_dims(t, axis=1))(inputs)
    x = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(x)
    mha = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    mha = LayerNormalization()(mha + x)  
    mha = GlobalAveragePooling1D()(mha)
    ffn = Dense(512, activation="relu")(mha)
    ffn = Dense(64, activation="relu")(ffn)
    ffn = Dense(output_dim, activation="linear")(ffn)
    model = keras.Model(inputs, ffn)  
    return model
```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8060745%2Fff94c23e5aafd6df986cf467c6cf9254%2F_2025-06-03_223607_687.png?generation=1748961385687832&alt=media)

Position encoding is crucial in time series and transformers. Similarly, it's vital in geographical sequences, especially when order matters. I tried both sinusoidal positional encoding and learnable positional encoding, but found that directly adding positional encoding worked best.

It is worth noting that our model already has convolutional layers and convolutional layers can inherently possess positional information. So is it still necessary for us to use positional encoding? 

The answer is **yes** , it would be better if we add positional encoding.

And this isn't an exception for this competition. Jonas2017[^Jonas] and ISLAM2020[^ISLAM] both indicated that convolutional layers can also be added, and adding them will improve the performance. ISLAM2020[^ISLAM] also indicated that larger convolutional kernels may capture more positional information.

This is exactly why I used a 7×7 convolutional kernel. Another reason is that in my practice, large-kernel convolutions can replace attention layers. And as is well known[^Simonyan], two 3×3 convolutional layers equal a 5×5 one spatially. Using larger kernels cuts layers.




## Ensemble
As mentioned earlier, I randomly selected 210,000 samples for each training session.

If we use two-thirds of the data for training each time, seven iterations ensure that 99.9% of the samples are trained at least once.

$1-(1/3)^6\approx 0.9986$

$1-(1/3)^7\approx 0.9995$



So:

final = (version 42+version 43+...+version 48)/7

## Shortcomings and Prospects

I didn't explore more models like LSTM and GRU, nor did I try larger models. 

Since positions aren't equidistant but positional encoding is, more innovative positional encoding methods can be further explored.

## References

[^Jonas]:Convolutional Sequence to Sequence Learning. https://arxiv.org/abs/1705.03122

[^ISLAM]:How Much Position Information Do Convolutional Neural Networks Encode. https://arxiv.org/abs/2001.08248

[^Simonyan]:Very deep convolutional networks for large-scale image recognition. https://arxiv.org/abs/1409.1556

