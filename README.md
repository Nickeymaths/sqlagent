# Table of content
1. [Fintuning](#finetuning)
    1. [Prepare data](#prepare-data)
    2. [Pre-processing](#pre-processing)
    3. [Load model](#load-model)

## Finetuning
Self-thinking
Finetuning LLM model is just Next Token Prediction training process where
- Structured input = Instruction + User input. IO format must conform to pretrained model's chat template
    - Llama 3.1 1B does not have to template so that we only need prompt as input
    - Examples:
      ```sh
      # Instruction
      You are a powerful text-to-SQL model. Your job is to answer questions about customers without addresses based on the provided SCHEMA. You must output the SQL query that answers the
      question. 
      # SCHEMA
      CREATE TABLE customer ( customer_key INT PRIMARY KEY ); 
      CREATE TABLE address ( address_key INT PRIMARY KEY, customer_key INT );
      # Input
      List customer keys of customers who have no address on file.
      # Response
      ```
- Structured output = Structured input + Response
                    = Response (need to be inserted structured input to head)
    - Examples:
      ```sh
      SELECT c.customer_key FROM customer c LEFT JOIN address a ON c.customer_key = a.customer_key WHERE a.customer_key IS NULL;
      ```
- Tokenize structured input words -> Tokens. Padding to max_length, padded tokens has attention mask equal zeros
    - Prompt: `Hey, are you conscious? Can you talk to me?`
    - Tokenized ids: `'input_ids': [128000, 19182, 11, 527, 499, 17371, 30, 3053, 499, 3137, 311, 757, 30], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
    - Decoded tokenized ids: `['<|begin_of_text|>', 'Hey', ',', ' are', ' you', ' conscious', '?', ' Can', ' you', ' talk', ' to', ' me', '?']`
- Freeze n first layers along with LoRA + quantization for trainable weight
- Froward and Backward -> Computational graph -> Compute gradient -> Update weight (in quantized version)
### Prepare data
- Download [spider](https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&authuser=0) and save to `data/raw/` folder
  - Train set: `data/raw/train_spider.json`, `data/raw/train_others.json`, total 8659 samples
  - Dev set: `data/raw/dev.json`, total 1034 samples
  - Test set: `data/raw/test.json`, total 2147 samples
- Data format in above json files: [see more](https://github.com/taoyds/spider). Each db_id is the name of db as folder name in `data/raw/database/<db_id>/` folder which contains sqlite, schema filder of database. We can schema file as schem for prompt.
- Input = `sample['question']`, Schema = `sample['db_id']`, Response = `sample['query']`
- Output
  - `data/raw/processed/train.csv`: contains `question`, `schema`, `query` that combines `train_spider.json`, `train_others.json`
  - `data/raw/processed/dev.csv`: same for `dev.json`
### Pre-processing
Zero-shot template
```
# Instruction
You are a power-full text-to-sql model. Your job is to answer question about a database. You are given question and context regarding on or more tables.
You only output the SQL query that answer the question.

### Input:
{input}

### Context:
{context}

### Response:
```

N-shot template
```
# Instruction
You are a power-full text-to-sql model. Your job is to answer question about a database. You are given question and context regarding one or more tables.
You only output the SQL query that answer the question.
First, I will show you few examples of question, context followed by correct SQL query.
Then, I will give you a new question and context, you should write the SQL query that appropiately completes the question.
...
### Example i Input: 
How many heads of the departments are older than 56 ?
### Example i Context:
These are tables such as department and head
Table department contains columns such as Department_ID: int, Name: text, Creation: text, Ranking: int, Budget_in_Billions: real, Num_Employees: real, Department_ID is primary key
Table head contains columns such as head_ID: int, name: text, born_state: text, age: real, head_ID is primary key
### Example i Response:
SELECT count(*) FROM head WHERE age  >  56
...
### New Input:
Show the years and the official names of the host cities of competitions.
### New Context:
There are tables such as city, farm_competition, competition_record
Table city contains columns such as City_ID: int, Official_Name: text, Status: text, Area_km_2: real, Population: real, Census_Ranking: text, City_ID is primary key
Table farm_competition contains columns such as Competition_ID: int, Year: int, Theme: text, Host_city_ID: int, Hosts: text, Competition_ID is primary key
Table competition_record contains columns such as Competition_ID: int, Farm_ID: int, Rank: int, ("Competition_ID","Farm_ID") is primary key
The Host_city_ID of farm_competition is foreign key of City_ID of city
The Competition_ID of competition_record is foreign key of Competition_ID of farm_competition
The Farm_ID of competition_record is foreign key of Farm_ID of farm
```
---
**NOTE**

In context section I prefer table, column description to sql schema. Using this format, the model is trained with can diverse contexts (only has neccessary table/columns, obfuscated with irrelevant table/columns). In real-world scenario, model should select important information from user input that need to answer the question. In few-shot version we need select related example from dataset (Random, Vector/Keyword search).

---
### Load model
### Training
### Evaluation and Model selection policy
### Save best model chekpoint

## Inference
### Load checkpoint
### Input formating
### Pre-processing
### Infer
### Post-processing

## Serving model
