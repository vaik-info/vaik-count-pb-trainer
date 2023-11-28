import tensorflow as tf

def prepare(class_num, image_size=320, bottle_neck=64, fine=False):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet' if fine else None,
                                                   include_top=False, input_shape=(image_size, image_size, 3))
    #x = base_model.layers[56].output
    x = base_model.layers[118].output
    #x = base_model.output
    x = tf.keras.layers.Conv2D(filters=bottle_neck, kernel_size=3, activation='relu', padding='same')(x)
    cam_output = tf.keras.layers.Conv2D(filters=class_num, kernel_size=3, activation='relu', padding='same')(x)
    predictions = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]))(cam_output)
    train_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    feature_extract_model = tf.keras.Model(inputs=base_model.input, outputs=x)
    save_model = tf.keras.Model(inputs=base_model.input, outputs=[predictions, cam_output])
    return train_model, feature_extract_model, save_model