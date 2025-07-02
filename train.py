from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.utils.quantization_config import  BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from loguru import logger

from config import Config, PROMPT_TEMPLATE


def run_train(dataset_dict: DatasetDict, config: Config):
    logger.info('Start training process')
    lora_config = load_lora_config(config)
    quantized_config = load_quantized_config(config)
    training_args = load_training_args(config)
    
    # Wrap quantization config
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    nf4_based_model = AutoModelForCausalLM.from_pretrained(config.model_id, quantization_config=quantized_config)
    nf4_based_model = prepare_model_for_kbit_training(nf4_based_model)
    
    # Wrap lora config
    peft_model = get_peft_model(nf4_based_model, lora_config)
    trainer = SFTTrainer(peft_model, 
                         args=training_args, 
                         train_dataset=dataset_dict['train'], 
                         eval_dataset=dataset_dict['validation'],
                         formatting_func=formatting_prompts_func)
    trainer.train()
    
    
def load_lora_config(config: Config):
    return LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias='none'
    )
    
def load_quantized_config(config: Config):
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_type),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant
    )

def load_training_args(config: Config):
    return TrainingArguments(
        output_dir=config.ckpt_dir,
        learning_rate=config.lr,
        optim=config.optimizer,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )

def load_model(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    return tokenizer, model

def formatting_prompts_func(example):
    return PROMPT_TEMPLATE.format(input=example['input'], context=example['context'], response=example['response'])

def data_collator_fn(examples):
    pass
