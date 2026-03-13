from tensorflow import keras

model = keras.models.load_model("models/cond_autoencoder.keras")
model.save("models/cond_autoencoder.h5")