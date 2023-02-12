import tensorflow as tf
def CNN1d(input_ecg):
    layer1=tf.keras.layers.Conv1D(filters=128,kernel_size=50,strides=3,padding='same',activation=tf.nn.relu)(input_ecg)
    BachNorm=tf.keras.layers.BatchNormalization()(layer1)
    MaxPooling1=tf.keras.layers.MaxPool1D(pool_size=2,strides=3)(BachNorm)
    layer2=tf.keras.layers.Conv1D(filters=32,kernel_size=7,strides=1,padding='same',activation=tf.nn.relu)(MaxPooling1)
    BachNorm = tf.keras.layers.BatchNormalization()(layer2)
    MaxPooling2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(BachNorm)
    layer3=tf.keras.layers.Conv1D(filters=32,kernel_size=10,strides=1,padding='same',activation=tf.nn.relu)(MaxPooling2)
    layer4=tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=2,padding='same',activation=tf.nn.relu)(layer3)
    MaxPooling3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(layer4)
    layer5 = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(MaxPooling3)
    layer6 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(layer5)
    flat=tf.keras.layers.Flatten()(layer6)
    x = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)(flat)
    x=tf.keras.layers.Dropout(rate=0.1)(x)
    label_ecg=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
    return label_ecg
