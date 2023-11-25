import tensorflow as tf
import glob
import os
import random
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
from data import ops


class CountDataset:
    classes = None
    colors = None
    image_dict = None
    output_signature = None
    min_size = None
    image_size = None
    aug_ratio = 0.5

    def __new__(cls, input_dir_path, classes, image_size=None, min_size=32):
        cls.classes = classes
        cls.image_size = image_size
        cls.image_dict_list = cls._prepare_image_dict(input_dir_path, classes)
        cls.min_size = min_size
        cls.output_signature = (
            tf.TensorSpec(name=f'raw_image', shape=(image_size, image_size, 3), dtype=tf.uint8),
            tf.TensorSpec(name=f'count', shape=(len(cls.classes), ), dtype=tf.float32)
        )
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=cls.output_signature
        )
        return dataset

    @classmethod
    def _generator(cls):
        while True:
            raw_image_path, count = random.choice(cls.image_dict_list)
            np_raw_image = tf.image.decode_image(tf.io.read_file(raw_image_path), channels=3).numpy()
            if random.uniform(0.0, 1.0) < cls.aug_ratio:
                np_raw_image = cls._data_aug(np_raw_image)
            np_raw_image = ops.resize_and_pad(np_raw_image, cls.image_size)
            if np_raw_image.shape[0] < cls.min_size or np_raw_image.shape[1] < cls.min_size:
                print(f'Pass {raw_image_path} because too small')
                continue
            yield (
                tf.convert_to_tensor(np_raw_image.astype(np.uint8)),
                tf.convert_to_tensor(np.asarray(count).astype(np.float32))
            )

    @classmethod
    def _prepare_image_dict(cls, input_dir_path, classes):
        image_dict_list = []
        raw_image_path_list = glob.glob(os.path.join(input_dir_path, f'**/*_raw.png'), recursive=True)
        for raw_image_path in tqdm(raw_image_path_list, desc='_prepare_image_dict()'):
            count = [0, ] * len(classes)
            json_path = raw_image_path.replace('.png', '.json')
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
                for key in json_dict.keys():
                    count[classes.index(key)] = json_dict[key]
            image_dict_list.append([raw_image_path, count])
        return image_dict_list

    @classmethod
    def _data_aug(cls, raw_image: np.array, random_r_ratio=0.25):
        raw_image = ops.random_resize(raw_image)
        raw_image = ops.random_padding(raw_image)
        raw_image = ops.random_hsv(raw_image, random_ratio=random_r_ratio)
        return raw_image.astype(np.uint8)


class TestCountDataset(CountDataset):
    max_sample = None

    def __new__(cls, input_dir_path, classes, image_size=None, min_size=32, max_sample=100):
        cls.max_sample = max_sample
        return super(TestCountDataset, cls).__new__(cls, input_dir_path, classes, image_size, min_size)

    @classmethod
    def _generator(cls):
        image_index = 0
        while True:
            if image_index > cls.max_sample - 1:
                break
            image_index += 1
            raw_image_path, count = random.choice(cls.image_dict_list)
            np_raw_image = tf.image.decode_image(tf.io.read_file(raw_image_path), channels=3).numpy()
            if random.uniform(0.0, 1.0) < cls.aug_ratio:
                np_raw_image = cls._data_aug(np_raw_image)
            np_raw_image = ops.resize_and_pad(np_raw_image, cls.image_size)
            if np_raw_image.shape[0] < cls.min_size or np_raw_image.shape[1] < cls.min_size:
                print(f'Pass {raw_image_path} because too small')
                continue
            yield (
                tf.convert_to_tensor(np_raw_image.astype(np.uint8)),
                tf.convert_to_tensor(np.asarray(count).astype(np.float32))
            )

    @classmethod
    def get_all_data(cls, dataset):
        dataset = iter(dataset)
        raw_image_list, count_list = [], []
        for data in dataset:
            raw_image_list.append(data[0])
            count_list.append(data[1])
        return np.stack(raw_image_list), np.stack(count_list)