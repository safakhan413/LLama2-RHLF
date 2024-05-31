import random
import numpy as np
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# Reduce batch size and max input length
train_batch_size = 4  # Further reduced batch size
gradient_accumulation_steps = 4  # Increase accumulation steps to maintain effective batch size
learning_rate = 1e-5
eval_batch_size = 1
eval_steps = 500
max_input_length = 256  # Reduce sequence length
save_steps = 1000
num_train_epochs = 20
random.seed(42)
DATA_PATH = "data/test-00000-of-00001-59ffb27399371eac.parquet"

df = pd.read_parquet(DATA_PATH)

from datasets import load_dataset
import json
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=256):
        self.post_list = []
        dataset = pd.read_parquet(train_path)
        self.labels = []
        for sample in dataset.iterrows():
            self.post_list.append(sample[1]["prompt"])
            self.labels.append(sample[1]["label"])

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        label = self.labels[idx]

        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        encodings_dict_label = self.tokenizer(label, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])
        labels_ids = torch.tensor(encodings_dict_label["input_ids"])
        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels_ids,
        }

print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.device_count())  # Check the number of available CUDA devices

tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
model = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py", use_cache=False).to("cuda:0")
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id

data_path = "data/test-00000-of-00001-59ffb27399371eac.parquet"
train_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "train",
    max_length=max_input_length,
)
for i in train_dataset:
    print(i["input_ids"], i["labels"])
    break

torch.cuda.set_device(0)
torch.cuda.empty_cache()  # Clear GPU cache

# Prepare the trainer and start training
training_args = TrainingArguments(
    output_dir='supervised-summarize-checkpoint',
    learning_rate=1e-5,
    per_device_train_batch_size=train_batch_size,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=2,
    warmup_steps=100,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
trainer.save_model("summarization_policy_new/")   ##path to save policy model

# Correctly handle max_length in text generation
model = AutoModelForCausalLM.from_pretrained("summarization_policy_new/")
model_path = "bigcode/tiny_starcoder_py"

tokenizer = AutoTokenizer.from_pretrained(model_path)
text = df.iloc[2]["prompt"]
tokenized_text = tokenizer(text, return_tensors="pt", max_length=256)

# Fix the generation step by setting max_new_tokens
generated_ids = model.generate(**tokenized_text, max_new_tokens=50)
decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Tokenizer decodes text on 0th location:", decoded_text)

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
