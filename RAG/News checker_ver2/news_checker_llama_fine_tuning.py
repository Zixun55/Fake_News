import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, PeftModel

# 設定模型名稱
model_name = 'meta-llama/Llama-2-7b-chat-hf'

# 量化配置（4-bit 量化）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 確保有 padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else '[PAD]'

# 載入 LLaMA-2 模型（2分類）
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    quantization_config=quantization_config,
    device_map='auto'
)
model.config.pad_token_id = tokenizer.pad_token_id

# 設定 LoRA 參數
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],  # 只對注意力機制進行 LoRA 調整
    lora_dropout=0.05,
    bias='none',
    task_type='SEQ_CLS'
)

# 加入 LoRA 微調
model = get_peft_model(model, lora_config)

# 讀取資料集
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# 轉換為 Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['Claim'], 
        padding='max_length',
        truncation=True,
        max_length=512
    )

# 應用 tokenizer
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 重新命名標籤列
train_dataset = train_dataset.rename_column('Classification', 'labels')
test_dataset = test_dataset.rename_column('Classification', 'labels')

# 設定資料格式為 PyTorch tensor
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 設定 Training 參數
training_args = TrainingArguments(
    output_dir='./results_ver2',      
    num_train_epochs=3,          
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=500,             
    weight_decay=0.01,            
    logging_dir='./logs',        
    logging_steps=10,
    evaluation_strategy='epoch',  
    save_strategy='epoch',
    save_total_limit=2,
    report_to='none'
)

# 訓練器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 開始 fine-tuning
trainer.train()

# === LoRA 合併 & 儲存最終模型 ===
# 1. 重新載入 base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# 2. 載入已 fine-tuned 的 LoRA 模型
model = PeftModel.from_pretrained(base_model, "./fine_tuned_model_ver2")

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# 3. 合併 LoRA 權重
model = model.merge_and_unload()

# 4. 儲存最終合併後的模型
model.save_pretrained('./fine_tuned_model_ver2_merged')
tokenizer.save_pretrained('./fine_tuned_model_ver2_merged')