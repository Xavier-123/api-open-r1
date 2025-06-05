import os

# 回调接口
callback_url = os.environ.get("CALLBACK_URL", "")

# 任务id
task_id = os.environ.get("TASK_ID", "")


# 蒸馏数据参数 - 全部通过传参控制
# file_path = os.environ.get("FILE_PATH", "")
# output_dataset = os.environ.get("OUTPUT_DATASET", "")

# sft训练参数 - 通过环境变量控制
model_name_or_path = os.environ.get("MODEL_NAME_OR_PATH", "")                                           # 基础大模型目录
dataset_name = os.environ.get("DATASET_NAME", r"F:\inspur\GPU\api-open-r1\file_save\4567\distilled")    # 蒸馏数据集目录
do_eval = os.environ.get("DO_EVAL", False)                                                              # 是否评估
learning_rate = os.environ.get("LEARNING_RATE", 5.0e-05)                                                # 学习率
num_train_epochs = os.environ.get("NUM_TRAIN_EPOCHS", 3)                                                # 训练 epochs 数
model_output_dir = os.environ.get("MODEL_OUTPUT_DIR", r"F:\inspur\GPU\api-open-r1\output")              # 蒸馏模型保存目录
per_device_train_batch_size = os.environ.get("TRAIN_BATCH_SIZE", 4)                                     # 训练 batch size
per_device_eval_batch_size = os.environ.get("EVAL_BATCH_SIZE", 4)                                       # 评估 batch size


class SFT_DISTILLED_REQ:
    pass
