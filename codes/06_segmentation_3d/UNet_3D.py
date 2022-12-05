import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras import losses, optimizers


def dice_coef(y_true, y_pred, smooth=1.):
    K = tf.keras.backend
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def dice_bce_loss(y_true, y_pred):
    BCE = losses.binary_crossentropy(y_true, y_pred)
    DCE = dice_coef_loss(y_true, y_pred)
    myLoss = 0.8*DCE + 0.2*BCE
    return myLoss


def ConvBN(x, filters):
    x = tf.keras.layers.Conv3D(filters, 3, padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    x = layers.Conv3D(filters*2, 3, padding='same',
                      activation=None, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x


def UpConvBN(x, A, filters):
    x = tf.keras.layers.concatenate([x, A], axis=-1)
    x = layers.Conv3DTranspose(
        filters, 3, padding='same', activation=None, strides=2)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    x = layers.Conv3D(filters, 3, padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x


def UNet_3D(img_size, d=1):

    inLayer = Input(shape=(*img_size, 1))

    # Analysis part
    A1 = ConvBN(inLayer, filters=int(16/d))
    A2 = ConvBN(A1, filters=int(32/d))
    A3 = ConvBN(A2, filters=int(64/d))

    # Bottleneck part
    x = layers.Conv3D(128, 3, padding='same', activation='relu')(A3)
    x = layers.Conv3D(128, 3, padding='same', activation='relu')(x)

    # Synthesis (decoding) part
    S1 = UpConvBN(x, A3, filters=int(64/d))
    S2 = UpConvBN(S1, A2, filters=int(32/d))
    S3 = UpConvBN(S2, A1, filters=int(16/d))

    outLayer = layers.Conv3D(1, 1, padding='same', activation='sigmoid')(S3)

    model = tf.keras.Model(inputs=inLayer, outputs=outLayer)

    model.compile(optimizer=optimizers.Adam(),
                  loss=dice_bce_loss, metrics=['accuracy', dice_coef])

    return model


if __name__ == '__main__':
    img_size = (128, 128, 128)
    model = UNet_3D(img_size, d=2)
    model.summary()
