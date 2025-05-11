from dataclasses import dataclass, field
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification # type: ignore
from transformers.data.data_collator import DataCollatorWithPadding
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
import wandb
import torch

wandb.login()

@dataclass
class DataArguments:
    model_path: str = field(metadata={"help": "The model path to use when running prediction."})
    max_train_samples: int = field(default=-1, metadata={"help": "Number of samples to be used during training or−1 if all training samplesshould be used. If a number n ̸=−1 is specified, you should select the first n samples in the training set."})
    max_eval_samples: int = field(default=-1, metadata={"help": "Number of samples to be used during evaluation or−1 if all evaluation samplesshould be used. If a number n ̸=−1 is specified, you should select the first n samples in the evaluation set."})
    max_predict_samples: int = field(default=-1, metadata={"help": "Number of samples to be used during prediction or−1 if all prediction samplesshould be used. If a number n ̸=−1 is specified, you should select the first n samples in the prediction set."})

@dataclass
class AdditionalTrainingArguments:
    lr: float = field(metadata={"help": "Learning rate for training."})
    batch_size: int = field(default=16, metadata={"help": "Batch size for training."})
    

def build_dataset(split: str, max_samples: int, dataset: DatasetDict):
    if max_samples > 0:
        sub_set = dataset[split].select(range(max_samples))
    else:
        sub_set = dataset[split]
    return sub_set

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}


def train(
        data_args: DataArguments,
        training_args: TrainingArguments,
        dataset: DatasetDict
        ):
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    def preprocess_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    train_dataset = build_dataset("train", data_args.max_train_samples, dataset).map(preprocess_function, batched=True)
    eval_dataset = build_dataset("validation", data_args.max_eval_samples, dataset).map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(data_args.model_path)

    eval_metrics = trainer.evaluate()

    return eval_metrics["eval_accuracy"]

def predict(
        data_args: DataArguments,
        dataset: DatasetDict
        ):
    
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_path)
    def preprocess_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, return_tensors="pt")
        
    model = AutoModelForSequenceClassification.from_pretrained(data_args.model_path)

    test_set: Dataset = dataset["test"]
    if data_args.max_predict_samples > 0:
        test_set = test_set.select(range(data_args.max_predict_samples))

    model.eval()

    str_to_write = ""

    with torch.no_grad():
        for sample in test_set:
            model_inputs = preprocess_function(sample)
            outputs = model(**model_inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            str_to_write += f"{sample['sentence1']}###{sample['sentence2']}###{pred}\n" # type: ignore

    with open("predictions.txt", "w") as f:
        f.write(str_to_write)


def main():


    parser = HfArgumentParser((TrainingArguments, DataArguments, AdditionalTrainingArguments)) # type: ignore
    training_args, data_args, additional_training_args = parser.parse_args_into_dataclasses()
    wandb.init(project="ex1",
               name=f"batch_size_{additional_training_args.batch_size}_lr_{additional_training_args.lr}",
               config = {
                "batch_size": additional_training_args.batch_size,
                "lr": additional_training_args.lr
               })

    training_args.per_device_train_batch_size = additional_training_args.batch_size
    training_args.learning_rate = additional_training_args.lr
    training_args.report_to = "wandb"
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 1

    dataset = load_dataset("nyu-mll/glue", "mrpc")

    if training_args.do_train:
        eval_accuracy = train(data_args, training_args, dataset) # type: ignore

        with open("res.txt", "a") as f:
            f.write(f"epoch_num : {training_args.num_train_epochs}, lr : {additional_training_args.lr}, batch_size : {additional_training_args.batch_size}, eval_accuracy : {eval_accuracy}\n")
                    
    if training_args.do_predict:
        predict(data_args, dataset) # type: ignore

    wandb.finish()


if __name__ == "__main__":
    main()








