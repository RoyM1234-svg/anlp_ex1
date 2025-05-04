from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification # type: ignore
from datasets import load_dataset, DatasetDict, Dataset 


@dataclass
class DataArguments:
    model_path: str = field(metadata={"help": "The model path to use when running prediction."})
    max_train_samples: int = field(default=-1, metadata={"help": "Number of samples to be used during training or−1 if all training samplesshould be used. If a number n ̸=−1 is specified, you should select the first n samples in the training set."})
    max_eval_samples: int = field(default=-1, metadata={"help": "Number of samples to be used during evaluation or−1 if all evaluation samplesshould be used. If a number n ̸=−1 is specified, you should select the first n samples in the evaluation set."})
    max_predict_samples: int = field(default=-1, metadata={"help": "Number of samples to be used during prediction or−1 if all prediction samplesshould be used. If a number n ̸=−1 is specified, you should select the first n samples in the prediction set."})

@dataclass
class AdditionalTrainingArguments:
    batch_size: int = field(default=16, metadata={"help": "Batch size for training."})
    lr: float = field(, metadata={"help": "Learning rate for training."})
    

def build_dataset(split: str, max_samples: int, dataset: DatasetDict):
    if max_samples > 0:
        sub_set = dataset[split].select(range(max_samples))
    else:
        sub_set = dataset[split]
    return sub_set

def train(data_args: DataArguments, training_args: TrainingArguments, dataset: DatasetDict):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def preprocess_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
    
    train_dataset = build_dataset("train", data_args.max_train_samples, dataset).map(preprocess_function, batched=True)
    eval_dataset = build_dataset("validation", data_args.max_eval_samples, dataset).map(preprocess_function, batched=True)

    print(train_dataset[0])

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

    



def predict():
    pass

def main():
    parser = HfArgumentParser((TrainingArguments, DataArguments, AdditionalTrainingArguments)) # type: ignore
    training_args, data_args, additional_training_args = parser.parse_args_into_dataclasses()

    training_args.per_device_train_batch_size = additional_training_args.batch_size

    dataset = load_dataset("nyu-mll/glue", "mrpc")
    
    if training_args.do_train:
        train(data_args, training_args, dataset) # type: ignore

    if training_args.do_predict:
        predict()


if __name__ == "__main__":
    main()








