from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import pandas as pd
from datasets import Dataset
from transformers import BitsAndBytesConfig

model_name = 'meta-llama/Llama-2-7b-hf'

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    quantization_config=quantization_config,
    device_map='auto'
)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='SEQ_CLS'
)

model = get_peft_model(model, lora_config)

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

def tokenize_function(examples):
    return tokenizer(
        examples['Claim'], 
        padding='max_length',
        truncation=True,
        max_length=512
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column('Classification', 'labels')
test_dataset = test_dataset.rename_column('Classification', 'labels')

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir='./results',      
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')