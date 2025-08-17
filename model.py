# model.py
# contains get_model(maxlen, vocab_size, embed_dim)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, BatchNormalization

def get_model(maxlen=700, vocab_size=21, embed_dim=128, conv_filters=64, lstm_units=64, dropout=0.4):
    inp = Input(shape=(maxlen,), name="seq_input")
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, mask_zero=False)(inp)
    x = Conv1D(filters=conv_filters, kernel_size=7, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout)(x)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    out = TimeDistributed(Dense(3, activation="softmax"), name="sse_output")(x)  # 3 classes per residue

    model = Model(inp, out)
    return model
