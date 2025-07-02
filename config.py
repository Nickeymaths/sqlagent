from dotenv import find_dotenv
from pydantic import (
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    NoDecode
)
from typing import Annotated

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_dotenv(), env_file_encoding='utf-8', extra='ignore')

    dataset_name: str
    data_dir: str
    processed_dir: str
    ckpt_dir: str
    log_file: str

    # Training
    model_id: str
    max_token: int
    batch_size: int
    gradient_checkpointing: bool
    optimizer: str
    gradient_accumulation_steps: int
    paged_optimization: bool
    lr: float
    r: int

    # Infer hyper-params
    kv_cache: bool

    # Quantization hyper-params
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_type: str
    bnb_4bit_use_double_quant: bool

    # LoRA hyper-params
    lora_alpha: int
    lora_dropout: float
    temperature: float
    target_modules: Annotated[list[str], NoDecode]

    @field_validator('target_modules', mode='before')
    @classmethod
    def decode_target_modules(cls, v: str) -> list[str]:
        return v.split(',')


EXTRACT_CREATE_STATEMENT_REGEX = r'create table\s+(?:if not exists\s*)?(?:.+\s+)+?\);?'
EXTRACT_TABLE_REGEX = r'create table\s+(?:if not exists\s*)?((?#tab)[^\s,]+)'
EXTRACT_COLUMN_REGEX = r'(?#col)([^\s,]+)\s+(?#type)(INTEGER|INT|TINYINT|SMALLINT|MEDIUMINT|BIGINT|UNSIGNED|INT2|INT8|CHARACTER|VARCHAR|VARYING|NCHAR|NATIVE|NVARCHAR|TEXT|CLOB|BLOB|REAL|DOUBLE|DOUBLE|FLOAT|NUMERIC|DECIMAL|BOOLEAN|DATE|DATETIME)'
EXTRACT_PRIMARY_KEY_REGEX = r'(?#pk)([^\s,]+)\s+(?:INTEGER|INT|TINYINT|SMALLINT|MEDIUMINT|BIGINT|UNSIGNED|INT2|INT8|CHARACTER|VARCHAR|VARYING|NCHAR|NATIVE|NVARCHAR|TEXT|CLOB|BLOB|REAL|DOUBLE|DOUBLE|FLOAT|NUMERIC|DECIMAL|BOOLEAN|DATE|DATETIME)\s+(?:[^\s,]+\s+)*primary key\s*,|primary key\s*\((?#pk)(.+)\),?'
EXTRACT_FOREIGN_KEY_REGEX = r'foreign key\s*\((?#pk)(.+)\)\s*references\s+(?#ref_col)(.+)\((?#ref_tab)(.+)\),?'
PROMPT_TEMPLATE = '''
# Instruction
You are a power-full text-to-sql model. Your job is to answer question about a database. You are given question and context regarding on or more tables.
You only output the SQL query that answer the question.

### Input:
{input}

### Context:
{context}

### Response:
{response}
'''
