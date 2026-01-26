import torch
import json
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    ViTImageProcessor, 
    BertTokenizer,
    Trainer, 
    TrainingArguments, 
    AutoImageProcessor, 
    AutoTokenizer,
    default_data_collator,
    EarlyStoppingCallback
)
from PIL import Image
import os
from typing import List, Dict

# 假设你的配置类已经定义好
from config.settings import settings, TrOCRTrainingConfig

class TrOCRDataset(torch.utils.data.Dataset):
    def __init__(self, feature_extractor, tokenizer, data_list: List[Dict]):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # 1. 加载图像
        image = Image.open(item["image_path"]).convert("RGB")
        
        # 2. 处理图像 (Pixel Values)
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.squeeze()
        
        # 3. 处理标签 (Labels)
        # 注意：text 不需要再次 squeeze，tokenizer 直接处理字符串
        labels = self.tokenizer(
            item["text"], 
            padding="max_length", 
            max_length=16, # 极验通常 3-5 个字，16 足够了
            truncation=True
        ).input_ids
        
        # 将 pad_token_id 替换为 -100 以忽略 loss 计算
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def train_trocr_model(config: TrOCRTrainingConfig, dataset_dir: str, resume: bool=False):
    print(f"--- 正在初始化 TrOCR 训练 ---")
    output_dir = os.path.join(settings.paths.base_dir, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    feature_extractor = ViTImageProcessor.from_pretrained(config.encoder_name)
    tokenizer = BertTokenizer.from_pretrained(config.decoder_name)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=config.encoder_name,
        decoder_pretrained_model_name_or_path=config.decoder_name
    )
    # 关键配置：将 BERT 的 Token ID 映射给 TrOCR
    model.config.decoder_start_token_id = tokenizer.cls_token_id # 中文多用 CLS
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # 生成参数配置
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 16 # 极验验证码长度
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    print("模型组装完成！Encoder: ViT, Decoder: BERT-Chinese")
    # 2. 从 JSONL 加载数据对
    def load_jsonl_data(jsonl_path, img_dir):
        dataset = []
        if not os.path.exists(jsonl_path):
            print(f"错误: 找不到标签文件 {jsonl_path}")
            return dataset
            
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                full_img_path = os.path.join(img_dir, data["file_name"])
                if os.path.exists(full_img_path):
                    dataset.append({
                        "image_path": full_img_path,
                        "text": data["text"]
                    })
        return dataset

    # 路径根据你的 settings 配置
    train_jsonl = os.path.join(dataset_dir, "labels.jsonl")
    train_img_dir = os.path.join(dataset_dir, "images")
    
    all_data = load_jsonl_data(train_jsonl, train_img_dir)
    
    # 简单切分训练集和验证集 (9:1)
    split = int(len(all_data) * 0.9)
    train_list = all_data[:split]
    val_list = all_data[split:]

    train_dataset = TrOCRDataset(feature_extractor, tokenizer, train_list)
    val_dataset = TrOCRDataset(feature_extractor, tokenizer, val_list)

    print(f"数据加载完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}")


    # 3. 训练参数设置
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        fp16=torch.cuda.is_available(), # 开启半精度加速
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True, # 训练结束时，自动加载效果最好的模型
        remove_unused_columns=False, # 必须设为 False 否则 pixel_values 会被删掉
        push_to_hub=False,
        metric_for_best_model=config.metric_for_best_model # 监控指标：验证集 Loss
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)], # 如果 eval_loss 连续 5 次评估（这里是 5 个 epoch）都没有下降，就停止训练
    )
    # 断点续练逻辑
    checkpoint = None
    # 如果输出目录存在，尝试寻找最新的 checkpoint
    if resume:
        if os.path.isdir(output_dir) and any("checkpoint-" in d for d in os.listdir(output_dir)):
            checkpoint = True # 设为 True，Trainer 会自动寻找最新的
            print(f"--- 检测到已存在的检查点，将从最新的 Checkpoint 恢复训练 ---")
    trainer.train(resume_from_checkpoint=checkpoint)

    # 6. 保存
    final_path = os.path.join(settings.paths.base_dir, config.output_dir, "final_trocr_model")
    trainer.save_model(final_path)
    feature_extractor.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"模型已保存至: {final_path}")