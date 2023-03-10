import tensorflow as tf
def CNN_LSTM(input_ecg):
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same', activation=tf.nn.relu)(
        input_ecg)
    #x=tf.keras.layers.Conv1D(filters=128, kernel_size=20, strides=3, padding='same',activation=tf.nn.relu)(input_ecg)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
    x=tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x=tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(x)
    x=tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x=tf.keras.layers.LSTM(10)(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dropout(rate=0.1)(x)
    x=tf.keras.layers.Dense(units=20, activation=tf.nn.relu)(x)
    x=tf.keras.layers.Dense(units=10, activation=tf.nn.relu)(x)
    x=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
    return x
