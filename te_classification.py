import autocuda
import torch
from metric_visualizer import MetricVisualizer
from transformers import AutoConfig

from omnigenome import (
    ClassificationMetric,
    OmniBPETokenizer,
    OmniGenomeModelForSequenceClassification,
    OmniGenomeTokenizer,
)
from omnigenome import OmniGenomeDatasetForSequenceClassification
from omnigenome import OmniSingleNucleotideTokenizer, OmniKmersTokenizer
from omnigenome import Trainer

label2id = {"0": 0, "1": 1}


class TEClassificationDataset(OmniGenomeDatasetForSequenceClassification):
    def __init__(self, data_source, tokenizer, max_length, **kwargs):
        super().__init__(data_source, tokenizer, max_length, **kwargs)

    def prepare_input(self, instance, **kwargs):
        sequence, labels = instance["sequence"].split("$LABEL$")
        sequence = sequence.strip()
        tokenized_inputs = self.tokenizer(
            sequence,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            tokenized_inputs["labels"] = self.label2id.get(str(labels), -100)
            tokenized_inputs["labels"] = torch.tensor(tokenized_inputs["labels"])
        return tokenized_inputs


epochs = 10
# epochs = 1
learning_rate = 1e-5
weight_decay = 1e-5
batch_size = 4
seeds = [42]
# seeds = [45, 46, 47]

compute_metrics = [
    ClassificationMetric(ignore_y=-100).accuracy_score,
    ClassificationMetric(ignore_y=-100, average="macro").f1_score,
]

mv = MetricVisualizer("Rice")

for gfm in [
    "yangheng/PlantRNA-FM",
]:
    for seed in seeds:
        train_file = "TE_Classification/Rice/train.txt"
        test_file = "TE_Classification/Rice/test.txt"
        valid_file = "TE_Classification/Rice/valid.txt"

        tokenizer = OmniGenomeTokenizer.from_pretrained(gfm, trust_remote_code=True)

        train_set = TEClassificationDataset(
            data_source=train_file,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=512,
        )
        test_set = TEClassificationDataset(
            data_source=test_file,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=512,
        )
        valid_set = TEClassificationDataset(
            data_source=valid_file,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=512,
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

        config = AutoConfig.from_pretrained(
            gfm, num_labels=len(label2id), trust_remote_code=True
        )

        ssp_model = OmniGenomeModelForSequenceClassification(
            gfm,
            tokenizer=tokenizer,
            label2id=label2id,
            trust_remote_code=True,
        )

        ssp_model.to(autocuda.auto_cuda())

        optimizer = torch.optim.AdamW(
            ssp_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        trainer = Trainer(
            model=ssp_model,
            train_loader=train_loader,
            eval_loader=valid_loader,
            test_loader=test_loader,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            compute_metrics=compute_metrics,
            seeds=seed,
            device=autocuda.auto_cuda(),
        )

        metrics = trainer.train()
        mv.log(gfm.split("/")[-1], "Accuracy", metrics["test"][-1]["accuracy_score"])
        mv.log(gfm.split("/")[-1], "F1", metrics["test"][-1]["f1_score"])
        # ssp_model.save("OmniGenome-185M", overwrite=True)
        print(metrics)
        mv.summary()
