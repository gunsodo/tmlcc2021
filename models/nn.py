import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# import tensorflow.compat.v2 as tf
# import tensorflow_datasets as tfds

# from tensorflow.python.framework.ops import disable_eager_execution

class NeuralNetwork:
    def __init__(self, dropout_rate=0.2):
        # tf.enable_v2_behavior()
        # disable_eager_execution()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(12, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        self.model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])

    def fit(self, X, y, epochs=150, batch_ratio=0.05):
        X = X.to_numpy()
        y = y.to_numpy()
        self.model.fit(X, y, epochs=epochs, batch_size=int(batch_ratio*len(X)))
    
    def predict(self, X):
        return self.model.predict(X.to_numpy())