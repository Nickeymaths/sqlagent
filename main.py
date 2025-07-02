import json
import os
import random
import time

import requests
import ssl
from tqdm import tqdm
from config import Config
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from datasets import Split
from peft import LoraConfig, TaskType
from huggingface_hub import configure_http_backend
from loguru import logger
from sql_metadata import Parser

from train import run_train
from util import _build_context, parse_sql_schema_with_regex


def read_config():
    return Config()

def config_logger(config: Config):
    logger.add(config.log_file, rotation='5MB', retention=1)

def save_sql_schema_json(data_dir: Path, out_dir: Path):
    if out_dir.is_dir():
        logger.info('Nothing to do, sql schema had been converted to json')
        return
    
    os.makedirs(out_dir, exist_ok=True)
    pbar = tqdm(list(data_dir.iterdir()), desc=f'Convert {data_dir} schemas to json')
    for db_folder in pbar:
        db_id = db_folder.name
        pbar.set_description(f"Processing {db_id}")
        
        schema_file = next(db_folder.glob('*.sql'), None)
        if schema_file:
            json_schema = None
            with open(schema_file, 'r') as f:
                schema = f.read()
                json_schema = parse_sql_schema_with_regex(schema)
            
            if not json_schema: continue
            
            out_file = out_dir / f'{db_id}.json'
            with open(out_file, 'w') as f:
                json.dump(json_schema, f)
            
def build_dataset(config: Config):
    data_dir = Path(config.data_dir)

    # Check if this dataset is processed or not
    output_dir = Path(config.processed_dir) / config.dataset_name
    if output_dir.is_dir():
        logger.info(f'Do nothing. {config.dataset_name} has already processed')
        return

    train_spider_json = format_json_input(config, load_json(data_dir/'train_spider.json'))
    train_others_json = format_json_input(config, load_json(data_dir/'train_others.json'))
    train_json = train_spider_json + train_others_json
    dev_json = format_json_input(config, load_json(data_dir/'dev.json'))
    test_json = format_json_input(config, load_json(data_dir/'test.json'))

    train_set = Dataset.from_list(train_json, split=Split.TRAIN)
    dev_set = Dataset.from_list(dev_json, split=Split.VALIDATION)
    test_set = Dataset.from_list(test_json, split=Split.TEST)
    
    dataset_dict = DatasetDict({
        'train': train_set,
        'validation': dev_set,  # or 'dev' if you prefer
        'test': test_set
    })

    # Save
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir, num_proc=os.cpu_count())

def load_json(fp):
    file = open(fp, 'r')
    result = json.load(file)
    file.close()
    return result

def format_json_input(config: Config, data_json: list[dict]) -> list[dict]:
    formated_json = []
    for sample in tqdm(data_json):
        input = sample['question']
        context = build_context(config, sample['db_id'], sample['query'])
        response = sample['query']
        formated_json.append(
            {
                'input': input,
                'context': context,
                'response': response
            }
        )
    return formated_json

def build_context(config: Config, db_id: str, sql: str) -> str:
    schema_file = Path(config.processed_dir) / 'database_schema' / f'{db_id}.json'
    if not schema_file.exists():
        schema_file = Path(config.processed_dir) / 'test_database_schema' / f'{db_id}.json'
        if not schema_file.exists():
            return ''
    
    sql_statement = Parser(sql)
    f = open(schema_file, 'r')
    schema_as_json = json.load(f)
    f.close()
    
    # argument other database schema about 50%
    query_tables = set(map(lambda t: t.lower(), sql_statement.tables))
    schema_tables = set(map(lambda t: t.lower(), schema_as_json['tables']))
    other_tables = schema_tables - query_tables
    if other_tables:
        random.seed(42)
        argumented_tables = random.choices(list(other_tables), k=int(0.5*len(query_tables)))
        tables = set(list(query_tables) + argumented_tables)
    else:
        tables = query_tables
    
    columns = list(filter(lambda x: True if x['table'].lower() in tables else False, schema_as_json['columns']))
    pks = list(filter(lambda x: True if x['table'].lower() in tables else False, schema_as_json['primary_keys']))
    fks = list(filter(lambda x: True if x['table'].lower() in tables else False, schema_as_json['foreign_keys']))
    
    columns = sorted(columns, key=lambda x: x['table'])
    pks = sorted(pks, key=lambda x: x['table'])
    fks = sorted(fks, key=lambda x: x['table'])
    
    return _build_context(columns, pks, fks)
    
def get_schema_file(data_dir: Path, db_id: str):
    db_dir = data_dir / 'database' / db_id
    if db_dir.is_dir():
        return next(db_dir.glob('*.sql'), None)
    
    db_dir = data_dir / 'test_database' / db_id
    if db_dir.is_dir():
        return next(db_dir.glob('*.sql'), None)
    
    return None

def setup_proxy():
    os.environ['HTTP_PROXY'] = '192.168.5.8:3128'
    os.environ['HTTPS_PROXY'] = '192.168.5.8:3128'

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

def main(config: Config):
    data_dir = Path(config.processed_dir) / config.dataset_name
    dataset_dict = load_from_disk(data_dir)
    run_train(dataset_dict, config)

if __name__ == '__main__':
    config = read_config()
    config_logger(config)
    save_sql_schema_json(
        Path(config.data_dir) / 'database', 
        Path(config.processed_dir) / 'database_schema'
    )
    save_sql_schema_json(
        Path(config.data_dir) / 'test_database', 
        Path(config.processed_dir) / 'test_database_schema'
    )
    build_dataset(config)
    main(config)
