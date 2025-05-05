import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import evaluate
import logging

# Silence evaluate's cache messages
logging.getLogger("evaluate").setLevel(logging.ERROR)


def prepare_data(config, dataset_name, tokenizer):
    dataset = load_dataset("glue", dataset_name)
    dataset_cols = config['dataset_to_columns'][dataset_name]
    max_length = config['max_length']
    batch_size = config['batch_size']

    def _tokenize_function(examples):
        # Single sentence tasks
        if "sentence2" not in dataset_cols:
            return tokenizer(
                examples[dataset_cols["sentence"]], padding="max_length", truncation=True, max_length=max_length
            )
        # Sentence pair tasks
        else:
            return tokenizer(
                examples[dataset_cols["sentence1"]], examples[dataset_cols["sentence2"]],
                padding="max_length", truncation=True, max_length=max_length
            )

    tokenized_dataset = dataset.map(_tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in dataset["train"].column_names if col != "label"]
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"], batch_size=batch_size, shuffle=True
    )

    val_key = "validation" if dataset_name != "mnli" else "validation_matched"
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset[val_key], batch_size=batch_size
    )
    
    return train_dataloader, eval_dataloader, len(tokenized_dataset["train"])


@torch.no_grad()
def evaluate_model(model, eval_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        
        outputs = model(**batch)
        logits = outputs.logits
        
        predictions = torch.argmax(logits, dim=-1)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metric = evaluate.load("accuracy")
    result = metric.compute(predictions=all_preds, references=all_labels)
    
    return result
