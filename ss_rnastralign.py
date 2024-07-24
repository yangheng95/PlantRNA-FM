# -*- coding: utf-8 -*-
# file: ssp_training.py
# time: 20:09 13/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import random

import autocuda
import torch
from transformers import AutoTokenizer

from omnigenome.src.dataset.omnigenome_dataset import (
    OmniGenomeDatasetForTokenClassification,
)
from omnigenome.src.metric.classification_metric import ClassificationMetric
from omnigenome.src.model.classiifcation.model import (
    OmniGenomeModelForTokenClassification,
)
from omnigenome import OmniGenomeTokenizer
from omnigenome.src.trainer.trainer import Trainer

label2id = {"(": 0, ")": 1, ".": 2}

model_name_or_path = "yangheng/PlantRNA-FM"
SN_tokenizer = OmniGenomeTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True
)
# SN_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


model = OmniGenomeModelForTokenClassification(
    model_name_or_path,
    tokenizer=SN_tokenizer,
    label2id=label2id,
    trust_remote_code=True,
)

epochs = 5
learning_rate = 2e-5
weight_decay = 1e-5
batch_size = 4
seed = [random.randint(0, 1000) for _ in range(3)]

train_file = "rnastralign/RNAstralign_dl_train.json"
test_file = "rnastralign/RNAstralign_dl_test.json"
valid_file = "archive2/ArchiveII_dl_valid.json"

train_set = OmniGenomeDatasetForTokenClassification(
    data_source=train_file,
    tokenizer=SN_tokenizer,
    label2id=label2id,
    max_length=512,
    max_examples=8000,
)
test_set = OmniGenomeDatasetForTokenClassification(
    data_source=test_file,
    tokenizer=SN_tokenizer,
    label2id=label2id,
    max_length=512,
    max_examples=1000,
)
valid_set = OmniGenomeDatasetForTokenClassification(
    data_source=valid_file,
    tokenizer=SN_tokenizer,
    label2id=label2id,
    max_length=512,
    max_examples=1000,
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

compute_metrics = ClassificationMetric(ignore_y=-100, average="macro").f1_score
for s in seed:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=valid_loader,
        test_loader=test_loader,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        compute_metrics=compute_metrics,
        seed=s,
        device=autocuda.auto_cuda(),
    )

    metrics = trainer.train()
