from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
import evaluate
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
from private_transformers import PrivacyEngine
import os
from tqdm.auto import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune models with optional differential privacy')
parser.add_argument('--finetune_mode', type=str, default='full', choices=['full', 'last-layer'], 
                    help='Fine-tuning mode: "full" for all layers or "last-layer" for classifier only')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
# Differential Privacy arguments
parser.add_argument('--enable_dp', action='store_true', help='Enable differential privacy during training')
parser.add_argument('--target_epsilon', type=float, default=8.0, help='Target privacy budget epsilon for DP training')
parser.add_argument('--max_grad_norm', type=float, default=0.1, help='Maximum L2 norm of per-sample gradients for DP training')
# Dataset arguments
parser.add_argument('--datasets', nargs='+', default=["sst2", "qnli", "qqp", "mnli"],
                   choices=["sst2", "qnli", "qqp", "mnli"], 
                   help='List of datasets to train on, space separated (e.g., --datasets sst2 qnli)')
args = parser.parse_args()

# Model configuration
model_checkpoint = "prajjwal1/bert-tiny"
max_length = 128

batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
finetune_mode = args.finetune_mode
enable_dp = args.enable_dp

# DP configuration
dp_config = {
    'target_epsilon': args.target_epsilon,
    'max_grad_norm': args.max_grad_norm,
}

# Dataset configuration
datasets_to_train = args.datasets
dataset_to_columns = {
    "sst2": {"sentence": "sentence", "label": "label"},
    "qnli": {"sentence1": "question", "sentence2": "sentence", "label": "label"},
    "mnli": {"sentence1": "premise", "sentence2": "hypothesis", "label": "label"},
    "qqp": {"sentence1": "question1", "sentence2": "question2", "label": "label"}
}

# Define tokenization function
def tokenize_function(examples, task):
    # Handle different input formats for different tasks
    if task in ["sst2"]:
        # Single sentence tasks
        return tokenizer(
            examples[dataset_to_columns[task]["sentence"]],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    else:
        # Sentence pair tasks
        return tokenizer(
            examples[dataset_to_columns[task]["sentence1"]],
            examples[dataset_to_columns[task]["sentence2"]],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

# Evaluation function
@torch.no_grad()
def evaluate_model(model, eval_dataloader, device, num_labels):
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

    # Compute metrics
    metric = evaluate.load("accuracy")
    result = metric.compute(predictions=all_preds, references=all_labels)
    
    return result

# Main training function
def train_model_on_dataset(dataset_name):
    print(f">> Training on {dataset_name} dataset... (ε={dp_config['target_epsilon'] if enable_dp else '∞'}, δ={dp_config['max_grad_norm'] if enable_dp else '∞'})")
    print(f"   Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
    
    # Load dataset
    dataset = load_dataset("glue", dataset_name)
    
    # Determine number of labels for the dataset
    num_labels = 3 if dataset_name == "mnli" else 2
    
    # Load model with appropriate number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # If fine-tuning only the last layer, freeze all other parameters
    trainable_params = []
    if finetune_mode == 'last-layer':
        print(f"Fine-tuning mode: {finetune_mode} - Only training the classifier layer")
        # Freeze all layers except the classifier
        for name, param in model.named_parameters():
            if 'classifier' in name:
                trainable_params.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # Check and print trainable parameters
        trainable_param_count = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_param_count} ({trainable_param_count/total_params:.2%} of total)")
    else:
        print(f"Fine-tuning mode: {finetune_mode} - Training all parameters")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {total_params} (100% of total)")
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, dataset_name),
        batched=True
    )
    
    # Prepare datasets with correct format
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in dataset["train"].column_names if col != "label"]
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"], 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_key = "validation" if dataset_name != "mnli" else "validation_matched"
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset[val_key], 
        batch_size=batch_size
    )
    
    # Create optimizer manually - only with trainable parameters
    optimizer = AdamW(
        trainable_params,
        lr=learning_rate, 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Setup output directory
    output_dir = f"./results/{dataset_name}_{'dp_' if enable_dp else ''}{datetime.now().strftime('%y%m%d.%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize privacy engine if DP is enabled
    privacy_engine = None
    if enable_dp:
        # Add warning suppression for the FutureWarning about backward hooks
        import warnings
        warnings.filterwarnings(
            "ignore", 
            message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated"
        )
        
        privacy_engine = PrivacyEngine(
            model,
            batch_size=batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            sample_size=len(tokenized_dataset["train"]),
            epochs=epochs,  
            max_grad_norm=dp_config['max_grad_norm'],
            target_epsilon=dp_config['target_epsilon'],
        )
        privacy_engine.attach(optimizer)
    
    # Training loop
    best_accuracy = 0.0
    train_metrics = []
    
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Training loop
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            # Regular training
            if not enable_dp:
                # Standard training approach
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = F.cross_entropy(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
            else:
                # DP training approach following dp_training_example.py
                optimizer.zero_grad()
                outputs = model(**batch)
                logits = outputs.logits
                
                # Compute per-sample losses
                loss = F.cross_entropy(logits, labels, reduction="none")
                
                # Use optimizer.step with loss as keyword argument
                optimizer.step(loss=loss)
                
                # Calculate mean loss for reporting
                batch_loss = loss.mean().item()
            
            total_loss += batch_loss
            num_batches += 1
            progress_bar.set_postfix({"loss": batch_loss})

        # Evaluate after each epoch
        avg_loss = total_loss / num_batches
        eval_results = evaluate_model(model, eval_dataloader, device, num_labels)
        accuracy = eval_results['accuracy']
        print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_loss:.4f}, val acc: {accuracy:.4f}", end="")
        
        # Save metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "eval_accuracy": accuracy
        }
        train_metrics.append(epoch_metrics)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f" (New best accuracy!)", end="")
            # Save model
            model_path = f"{output_dir}/best_model"
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        print()
    
    # Save final model
    final_model_path = f"{output_dir}/final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model for {dataset_name} saved to {final_model_path}")
    
    # Final evaluation
    model.eval()
    final_eval_results = evaluate_model(model, eval_dataloader, device, num_labels)
    
    # Print final results
    print(f"\n===== {dataset_name.upper()} RESULTS =====")
    print(f"  Best accuracy: {best_accuracy:.4f}")
    print(f"  Final accuracy: {final_eval_results['accuracy']:.4f}")
    # print("  Training details:")
    # for metric in train_metrics:
    #     print(f"    Epoch {metric['epoch']}: loss={metric['train_loss']:.4f}, accuracy={metric['eval_accuracy']:.4f}")
    print("=" * 30)
    
    final_eval_results["best_accuracy"] = best_accuracy
    return final_eval_results


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Train on all specified datasets
results = {}
for dataset_name in datasets_to_train:
    results[dataset_name] = train_model_on_dataset(dataset_name)

# Print summary of results with DP information
print("\n===== SUMMARY OF RESULTS =====")
print(f"Training with Differential Privacy (ε={dp_config['target_epsilon'] if enable_dp else '∞'}, δ={dp_config['max_grad_norm'] if enable_dp else '∞'})")
# print args
print(f"Fine-tuning mode: {finetune_mode}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print("=" * 30)
for dataset_name, result in results.items():
    print(f"{dataset_name}: {result}")
