from PIL import Image
import os
import tensorflow as tf

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_model, feature_extract_model, save_model_dir_path, prefix, valid_data, train_valid_data, logging_sample=10):
        super(SaveCallback, self).__init__()
        os.makedirs(save_model_dir_path, exist_ok=True)

        self.save_model_dir_path = save_model_dir_path
        self.prefix = prefix
        self.valid_data = valid_data
        self.train_valid_data = train_valid_data
        self.logging_sample = logging_sample
        self.save_model = save_model
        self.feature_extract_model = feature_extract_model

    def on_epoch_end(self, epoch, logs=None):
        loss_string = "_".join([f'{k}_{v:.4f}' for k, v in logs.items()])
        save_model_name = f'{self.prefix}_epoch-{epoch}_{loss_string}'
        output_model_dir_path = os.path.join(self.save_model_dir_path, save_model_name)
        os.makedirs(output_model_dir_path, exist_ok=True)
        self.save_model.save(output_model_dir_path)

        output_model_dir_path += "_feature_ckpt"
        os.makedirs(output_model_dir_path, exist_ok=True)
        self.feature_extract_model.save_weights(os.path.join(output_model_dir_path, 'weight.ckpt'))

        self.predict_dump(output_model_dir_path + '_log_train', self.train_valid_data)
        self.predict_dump(output_model_dir_path + '_log_valid', self.valid_data)

    def predict_dump(self, output_log_dir_path, valid_data):
        os.makedirs(output_log_dir_path, exist_ok=True)
        result = self.model.predict(valid_data[0][:self.logging_sample])
        for index, image in enumerate(valid_data[0][:self.logging_sample]):
            answer_string = ""
            for answer in valid_data[1][index]:
                answer_string += f'{int(answer)}_'
            answer_string = answer_string[:-1]
            inf_string = ""
            for inf in result[index]:
                inf_string += f'{round(inf)}_'
            inf_string = inf_string[:-1]
            Image.fromarray(image).save(os.path.join(output_log_dir_path, f'{index:04d}_ans-{answer_string}_inf-{inf_string}.png'))