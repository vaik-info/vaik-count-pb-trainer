from PIL import Image
import os
import tensorflow as tf

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_model_dir_path, prefix, valid_data, logging_sample=10):
        super(SaveCallback, self).__init__()
        os.makedirs(save_model_dir_path, exist_ok=True)

        self.save_model_dir_path = save_model_dir_path
        self.prefix = prefix
        self.valid_data = valid_data
        self.logging_sample = logging_sample

    def on_epoch_end(self, epoch, logs=None):
        loss_string = "_".join([f'{k}_{v:.4f}' for k, v in logs.items()])
        save_model_name = f'{self.prefix}_epoch-{epoch}_{loss_string}'
        output_model_dir_path = os.path.join(self.save_model_dir_path, save_model_name)
        os.makedirs(output_model_dir_path, exist_ok=True)
        self.model.save(output_model_dir_path)

        output_log_dir_path = output_model_dir_path + '_log'
        self.predict_dump(output_log_dir_path)

    def predict_dump(self, output_log_dir_path):
        os.makedirs(output_log_dir_path, exist_ok=True)
        result = self.model.predict(self.valid_data[0][:self.logging_sample])
        for index, image in enumerate(self.valid_data[0][:self.logging_sample]):
            answer_string = ""
            for answer in self.valid_data[1][index]:
                answer_string += f'{int(answer)}_'
            answer_string = answer_string[:-1]
            inf_string = ""
            for inf in result[index]:
                inf_string += f'{inf:.2f}_'
            inf_string = inf_string[:-1]
            Image.fromarray(image).save(os.path.join(output_log_dir_path, f'{index:04d}_ans-{answer_string}_inf-{inf_string}.png'))