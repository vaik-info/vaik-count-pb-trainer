import tensorflow as tf

def prepare(class_num, image_size=(224, 224), bottle_neck=64, fine=False):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet' if fine else None,
                                                   include_top=False, input_shape=(image_size, image_size, 3))
    x = base_model.output
    cam_output = tf.keras.layers.Conv2D(filters=bottle_neck, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(cam_output)
    predictions = tf.keras.layers.Dense(class_num, activation='relu')(x)
    train_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    save_model = tf.keras.Model(inputs=base_model.input, outputs=[predictions, cam_output])
    return train_model, save_model