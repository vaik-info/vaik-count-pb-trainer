import tensorflow as tf

def prepare(class_num, image_size=(224, 224), bottle_neck=64, fine=False):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet' if fine else None,
                                                   include_top=False, input_shape=(image_size, image_size, 3))
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(bottle_neck, activation='relu')(x)
    predictions = tf.keras.layers.Dense(class_num)(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model