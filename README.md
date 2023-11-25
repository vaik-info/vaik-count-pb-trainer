# vaik-count-pb-trainer

Train count pb model

## train_pb.py

### Usage

```shell
pip install -r requirements.txt
python train_pb.py --train_input_dir_path ~/.vaik-mnist-count-dataset/train \
                --valid_input_dir_path ~/.vaik-mnist-count-dataset/valid \
                --classes_json_path ~/.vaik-mnist-count-dataset/classes.json \
                --model_type mobile_net_v2_cam_model \
                --epochs 20 \
                --step_size 1000 \
                --batch_size 16 \
                --test_max_sample 100 \
                --image_size 320 \
                --output_dir_path ~/.vaik-count-pb-trainer/output_model        
```

- train_input_dir_path & valid_input_dir_path

```shell
train/
├── train_000000000_raw.png
├── train_000000000_raw.json
├── train_000000001_raw.png
├── train_000000001_raw.json
├── train_000000002_raw.png
・・・
```

### Output

![count_pb_trainer01](https://github.com/vaik-info/vaik-count-pb-trainer/assets/116471878/d7711967-db7e-41a3-93e3-cc7506683eb5)
![count_pb_trainer02](https://github.com/vaik-info/vaik-count-pb-trainer/assets/116471878/2fe923e6-7b56-4a83-b97c-ca28c6d6a277)
![count_pb_trainer03](https://github.com/vaik-info/vaik-count-pb-trainer/assets/116471878/340991ab-e48f-477e-b57d-c94a8a5be796)
