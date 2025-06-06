import os
import yaml
import subprocess
import torch
from tools.log import logger
os.environ["WANDB_DISABLED"] = "true"        # 关闭连接hunggingface

os.environ.setdefault("TASK_ID", "1234")
os.environ.setdefault("ACCELERATE_TYPE", "zero3")        # ddp/fsdp/zero2/zero3
# os.environ.setdefault("MODEL_NAME_OR_PATH", r"F:\inspur\LLM_MODEL\Qwen\Qwen2___5-0___5B-Instruct-GPTQ-Int4")
os.environ.setdefault("MODEL_NAME_OR_PATH", r"/nfs_data/models/Qwen/Qwen2.5-0.5B-Instruct")
# os.environ.setdefault("DATASET_NAME", r"F:\inspur\GPU\api-open-r1\file_save\4567\distilled")
os.environ.setdefault("DATASET_NAME", r"/nfs_data/xaw/deploy/open-r1/api-open-r1/file_save/1234/distilled")
os.environ.setdefault("DO_EVAL", "False")
os.environ.setdefault("LEARNING_RATE", "5.0e-05")
os.environ.setdefault("NUM_TRAIN_EPOCHS", "1")
# os.environ.setdefault("OUTPUT_DIR", r"F:\inspur\GPU\api-open-r1\file_save\1234\output")
os.environ.setdefault("OUTPUT_DIR", r"/nfs_data/xaw/deploy/open-r1/api-open-r1/file_save/1234/output")
os.environ.setdefault("TRAIN_BATCH_SIZE", "4")
os.environ.setdefault("EVAL_BATCH_SIZE", "4")


# 使用SFT微调模型
def train():
    # task_id = os.environ.get("TASK_ID")
    # dir_path = os.path.split(os.path.abspath(__file__))[0] + f"/file_save/{task_id}"
    dir_path = os.path.split(os.path.abspath(__file__))[0] + f"/file_save"
    os.makedirs(dir_path, exist_ok=True)

    # 更改 accelerate 配置
    accelerate_type = os.environ.get("ACCELERATE_TYPE")
    accelerate_configs_path = os.path.join(os.path.abspath(os.getcwd()), f"file_save/{accelerate_type}.yaml")
    with open(accelerate_configs_path, 'r') as f:
        data = yaml.safe_load(f)
        data["num_processes"] = torch.cuda.device_count()

    logger.info("accelerate args:")
    logger.info(data)
    logger.info("")

    # with open(os.path.join(dir_path, "accelerate_configs.yaml"), 'w', encoding='utf-8') as f:
    with open(accelerate_configs_path, 'w', encoding='utf-8') as f:             # 使用容器启动可直接修改配置文件
        # 使用 yaml.dump() 方法将字典 d 转换为 YAML 格式，并将其写入文件中
        f.write(yaml.dump(data))

    # 根据环境变量，更改sft训练配置
    sft_configs_path = os.path.join(os.path.abspath(os.getcwd()), "file_save/sft_config_demo.yaml")
    with open(sft_configs_path, 'r') as f:
        data = yaml.safe_load(f)

        # 如果指定了dataset_name，则使用指定数据集；若未指定，则根据task_id获取蒸馏数据集
        if os.environ.get("DATASET_NAME") and len(os.environ.get("DATASET_NAME")) > 0:
            data["dataset_name"] = os.environ.get("DATASET_NAME")
        else:
            data["dataset_name"] = os.path.join(dir_path, "distilled")

        # 若未指定模型保存目录，则保存在对应task_id目录下的model_save目录下
        if os.environ.get("OUTPUT_DIR") and len(os.environ.get("OUTPUT_DIR")) > 0:
            data["output_dir"] = os.environ.get("OUTPUT_DIR")
        else:
            data["output_dir"] = os.path.join(dir_path, "output_dir")


        data["model_name_or_path"] = os.environ.get("MODEL_NAME_OR_PATH")              # 基础大模型
        data["learning_rate"] = os.environ.get("LEARNING_RATE")                        # 学习率
        data["num_train_epochs"] = os.environ.get("NUM_TRAIN_EPOCHS")                  # 训练 epochs 数
        data["per_device_train_batch_size"] = os.environ.get("TRAIN_BATCH_SIZE")       # 训练 batch size
        data["do_eval"] = os.environ.get("DO_EVAL")                                    # 控制是否评估
        data["per_device_eval_batch_size"] = os.environ.get("EVAL_BATCH_SIZE")         # 评估 batch size

    logger.info("sft args:")
    logger.info(data)
    logger.info("")

    # with open(os.path.join(dir_path, "sft_config_demo.yaml"), 'w', encoding='utf-8') as f:
    with open(sft_configs_path, 'w', encoding='utf-8') as f:
        f.write(yaml.dump(data))

    system_cmd_str_list = ["accelerate", "launch",
                           # "--config_file", os.path.abspath(os.getcwd()) + f"/file_save/{task_id}/accelerate_configs.yaml",
                           "--config_file", accelerate_configs_path,
                           os.path.join(os.path.abspath(os.getcwd()), "third_party_tools", "open-r1/src/open_r1/sft.py"),
                           # "--config", os.path.join(dir_path, "sft_configs.yaml")]
                           "--config", sft_configs_path]
    logger.info(" ".join(system_cmd_str_list))

    result = subprocess.run(system_cmd_str_list, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"成功！输出：{result.stdout}")
    else:
        logger.info(f"失败！输出：{result.stdout}")
        logger.info(f"失败！输出：{result.stderr}")

def main():
    train()


if __name__ == '__main__':
    main()