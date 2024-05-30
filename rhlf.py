# import random

# import numpy as np
# import torch
# import pandas as pd

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     Trainer,
#     TrainingArguments,
#     default_data_collator,
# )


# def set_seed(seed_val=42):
#     random.seed(seed_val)
#     np.random.seed(seed_val)
#     torch.manual_seed(seed_val)
#     torch.cuda.manual_seed_all(seed_val)


# train_batch_size = 8
# gradient_accumulation_steps = 2
# learning_rate = 1e-5
# eval_batch_size = 1
# eval_steps = 500
# max_input_length = 550
# save_steps = 1000
# num_train_epochs = 20
# random.seed(42)

# from datasets import load_dataset
# # df = pd.read_parquet("data/test-00000-of-00001-59ffb27399371eac.parquet") ## downloded above linked dataset,
# # print(df.head(5))
# import json

# import pandas as pd
# import torch
# from datasets import load_dataset
# from torch.utils.data import Dataset


# class TLDRDataset(Dataset):
#     def __init__(self, train_path, tokenizer, split, max_length=256):
#         self.post_list = []
#         dataset = pd.read_parquet(train_path)
#         self.labels = []
# #         dataset = dataset[:100]
#         for sample in dataset.iterrows():
#             self.post_list.append(sample[1]["prompt"])
#             self.labels.append(sample[1]["label"])

#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.input_ids = []
#         self.attn_masks = []

#     def __len__(self):
#         return len(self.post_list)

#     def __getitem__(self, idx):
#         txt = self.post_list[idx]
#         label = self.labels[idx]

#         encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
#         encodings_dict_label = self.tokenizer(label,truncation=True, max_length=self.max_length, padding="max_length")
#         input_ids = torch.tensor(encodings_dict["input_ids"])
#         attn_masks = torch.tensor(encodings_dict["attention_mask"])
#         labels_ids = torch.tensor(encodings_dict_label["input_ids"])
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attn_masks,
#             "labels": labels_ids,
#         }
# import torch

# print(torch.cuda.is_available())  # Check if CUDA is available
# print(torch.cuda.device_count())  # Check the number of available CUDA devices

# tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
# model = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py", use_cache=False).to("cuda:0")
# tokenizer.pad_token = tokenizer.eos_token
# model.resize_token_embeddings(len(tokenizer))
# tokenizer.pad_token_id = tokenizer.eos_token_id
# model.config.end_token_id = tokenizer.eos_token_id
# model.config.pad_token_id = model.config.eos_token_id
# print(model.config)

# data_path = "data/test-00000-of-00001-59ffb27399371eac.parquet"
# train_dataset = TLDRDataset(
#     data_path,
#     tokenizer,
#     "train",
#     max_length=256,
# )
# for i in train_dataset:
#     print(i["input_ids"], i["labels"])
#     print(i)
#     break

# torch.cuda.set_device(0)
# torch.cuda.empty_cache()

# # Prepare the trainer and start training
# training_args = TrainingArguments(
#     output_dir='supervised-summarize-checkpoint',
#     learning_rate=1e-5,
#     per_device_train_batch_size=8,
# #     per_device_eval_batch_size=eval_batch_size,
#     fp16=True,
#     gradient_accumulation_steps=2,
#     num_train_epochs=2,
#     warmup_steps=100,
#     logging_steps=10,
# )

# training_args.device.index

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
# #     compute_metrics=compute_metrics,
# #     data_collator=default_data_collator,
# #     preprocess_logits_for_metrics=preprocess_logits_for_metrics
# )
# trainer.train()
# # trainer.save_model(output_dir)

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
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("summarization_policy_new/")
model_path = "bigcode/tiny_starcoder_py"

tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, max_length=256, padding="max_length")
text = df.iloc[2]["prompt"]
tokenized_text = tokenizer(text, return_tensors="pt", max_length=256)
tokenizer.decode(model.generate(**tokenized_text)[0])

