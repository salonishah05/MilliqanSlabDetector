import os
import numpy as np
from tensorflow import keras
from keras import layers

ART_DIR = "artifacts"
X_train = np.load(os.path.join(ART_DIR, "X_train.npy")).astype("float32")
X_val   = np.load(os.path.join(ART_DIR, "X_val.npy")).astype("float32")
chan_train = np.load(os.path.join(ART_DIR, "chan_train.npy")).astype("int32")
chan_val   = np.load(os.path.join(ART_DIR, "chan_val.npy")).astype("int32")

P = X_train.shape[1]   
N_CHANNELS = int(max(chan_train.max(), chan_val.max())) + 1
print(f"P={P}, N_CHANNELS={N_CHANNELS}, N_train={len(X_train)}, N_val={len(X_val)}")

EMB_DIM = 13         
ENC_HIDDEN = 64
LATENT_DIM = 16
DEC_HIDDEN = 64
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 512
WEIGHT_DECAY = 1e-5     
PATIENCE = 15

feat_in = keras.Input(shape=(P,), name="features")
chan_in = keras.Input(shape=(), dtype="int32", name="chan_id")

chan_emb = layers.Embedding(input_dim=N_CHANNELS, output_dim=EMB_DIM, name="chan_embedding")(chan_in)
chan_emb = layers.Flatten()(chan_emb)

NOISE_STD = 0.02
noisy_feat = layers.GaussianNoise(NOISE_STD)(feat_in)
enc_in = layers.Concatenate(name="enc_concat")([noisy_feat, chan_emb])
x = layers.Dense(ENC_HIDDEN, activation="relu", kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY))(enc_in)
x = layers.Dense(ENC_HIDDEN//2, activation="relu", kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY))(x)
z = layers.Dense(LATENT_DIM, activation=None, name="latent")(x)

dec_in = layers.Concatenate(name="dec_concat")([z, chan_emb])
y = layers.Dense(DEC_HIDDEN//2, activation="relu", kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY))(dec_in)
y = layers.Dense(DEC_HIDDEN, activation="relu", kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY))(y)
out = layers.Dense(P, activation="linear", name="recon")(y) 

model = keras.Model(inputs=[feat_in, chan_in], outputs=out, name="cond_autoencoder")
model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
model.summary()


callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
]

history = model.fit(
    {"features": X_train, "chan_id": chan_train},
    X_train,
    validation_data=({"features": X_val, "chan_id": chan_val}, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)

os.makedirs("models", exist_ok=True)
model.save("models/cond_autoencoder.keras")

encoder = keras.Model(inputs=[feat_in, chan_in], outputs=model.get_layer("latent").output, name="encoder")
encoder.save("models/cond_encoder.keras")

train_recon = model.predict({"features": X_train, "chan_id": chan_train}, verbose=0)
train_score = np.mean((X_train - train_recon)**2, axis=1)

q = 0.995
thr_by_chan = np.full((N_CHANNELS,), np.nan, dtype=np.float32)

for c in range(N_CHANNELS):
    s = train_score[chan_train == c]
    if len(s):
        thr_by_chan[c] = np.quantile(s, q)

np.save(os.path.join(ART_DIR, "thr_by_chan.npy"), thr_by_chan)