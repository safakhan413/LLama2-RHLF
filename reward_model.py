#_________________________ Training the reward function

import torch
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from trl import RewardTrainer, SFTTrainer
from datasets import Dataset
import json
import pandas as pd
from transformers import Trainer, TrainingArguments

##model path
MODEL_PATH = "bigcode/tiny_starcoder_py"
DATA_PATH = "data/test-00000-of-00001-59ffb27399371eac.parquet"

df = pd.read_parquet(DATA_PATH)
df = df[:10]
raw_dataset = Dataset.from_pandas(df)
print(raw_dataset)

##defininig the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def formatting_func(examples):
    kwargs = {"padding": "max_length",
              "truncation": True,
              "max_length": 256,
              "return_tensors": "pt"
              }

    # Prepend the prompt and a line break to the original_response and response-1 fields.
    prompt_plus_chosen_response = examples["prompt"] + "\n" + examples["chosen"]
    prompt_plus_rejected_response = examples["prompt"] + "\n" + examples["rejected"]

    # Then tokenize these modified fields.
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }

formatted_dataset = raw_dataset.map(formatting_func)
formatted_dataset = formatted_dataset.train_test_split()

print(model.config)
training_args = TrainingArguments(
        output_dir="rm_checkpoint/",
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        learning_rate=1e-5,
        save_total_limit=1,
        no_cuda=True
    )

trainer = RewardTrainer(model=model,
                        tokenizer=tokenizer,
                        train_dataset=formatted_dataset['train'],
                        eval_dataset=formatted_dataset['test'],
                        args= training_args
                        )
trainer.train()

trainer.save_model("rm_model/")
## inference the model
rm_model = AutoModelForCausalLM.from_pretrained("rm_model/")
tokenizer = AutoTokenizer.from_pretrained("model/")
def get_score(model, tokenizer, prompt, response):

    instructions = tokenizer.encode_plus(prompt,
                                       response,
                                       padding="max_length",
                                       max_length=256,
                                       return_tensors="pt",
                                        truncation=True)
    with torch.no_grad():
        outputs = model(**instructions)

    logits = outputs[0]

    return logits
# usage with prompt
prompt = df.iloc[0]["prompt"]
example_prefered_response = df.iloc[0]["chosen"]
example_unprefered_response = df.iloc[0]["rejected"]

loss1 = get_score(model, tokenizer, prompt, example_prefered_response)
loss2= get_score(model, tokenizer, prompt, example_unprefered_response)
from torch import nn
loss = -nn.functional.logsigmoid(loss1 - loss2).mean()
print(tokenizer.decode(torch.max(loss1, axis=-1).indices[0]))
