from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
import os
from tqdm.auto import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import private_transformers.privacy_engine as pe_module
from utils import prepare_data, evaluate_model

# Silence evaluate's cache messages
import logging
logging.getLogger("evaluate").setLevel(logging.ERROR)


# Function to parse command line arguments and setup configuration
def setup_config():
    parser = argparse.ArgumentParser(description='Fine-tune models with optional differential privacy')
    parser.add_argument('--model_checkpoint', type=str, default="prajjwal1/bert-tiny", help='Base model checkpoint')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length for tokenizer')
    parser.add_argument('--finetune_mode', type=str, default='full',
                        choices=['full', 'last-layer', 'lora', 'ia3', 'prefix', 'soft-prompt', 'soft-prompt+lora', 'prefix+lora'],
                        help='Fine-tuning mode')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    # DP args
    parser.add_argument('--enable_dp', action='store_true', help='Enable differential privacy')
    parser.add_argument('--target_epsilon', type=float, default=8.0, help='Target epsilon for DP')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='Max grad norm for DP')
    # Dataset args
    parser.add_argument('--datasets', nargs='+', default=["sst2", "qnli", "qqp", "mnli"],
                       choices=["sst2", "qnli", "qqp", "mnli"],
                       help='List of datasets to train on')
    # LoRA args
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    # Prefix/Prompt args
    parser.add_argument('--num_prepend_tokens', type=int, default=10, help='Number of prepend tokens for Prefix/Soft-prompt')
    
    args = parser.parse_args()
    
    config = {
        'model_checkpoint': args.model_checkpoint,
        'max_length': args.max_length,
        'finetune_mode': args.finetune_mode,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'enable_dp': args.enable_dp,
        'dp_config': {
            'target_epsilon': args.target_epsilon,
            'max_grad_norm': args.max_grad_norm,
        },
        'datasets_to_train': args.datasets,
        'lora_config': {
            'r': args.lora_r,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
        },
        'prefix_prompt_config': {
            'num_prepend_tokens': args.num_prepend_tokens,
        },
        'dataset_to_columns': {
            "sst2": {"sentence": "sentence", "label": "label"},
            "qnli": {"sentence1": "question", "sentence2": "sentence", "label": "label"},
            "mnli": {"sentence1": "premise", "sentence2": "hypothesis", "label": "label"},
            "qqp": {"sentence1": "question1", "sentence2": "question2", "label": "label"}
        }
    }
    return config


def prepare_model(config, num_labels):
    # Suppress transformers warning about newly initialized weights
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_checkpoint'],
        num_labels=num_labels
    )
    
    # Freeze all parameters. Classification layer is always trainable
    for name, param in model.named_parameters():
        param.requires_grad = 'classifier' in name

    finetune_mode = config['finetune_mode']
    lora_config = config['lora_config']
    prefix_prompt_config = config['prefix_prompt_config']

    print(f"   Preparing model for fine-tuning mode: {finetune_mode}")

    if finetune_mode == 'full':
        print(f"   - Training all parameters")
        for param in model.parameters():
            param.requires_grad = True

    elif finetune_mode == 'last-layer':
        print(f"   - Only training the classifier layer")

    elif finetune_mode == 'lora':
        print(f"   - Using LoRA with rank={lora_config['r']}, alpha={lora_config['alpha']}, dropout={lora_config['dropout']}")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=lora_config['r'], lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'], bias="none", target_modules=["query", "key", "value"],
            modules_to_save=["classifier"]
        )
        model = get_peft_model(model, peft_config)

    elif finetune_mode == 'prefix':
        print(f"   - Using Prefix Tuning with {prefix_prompt_config['num_prepend_tokens']} tokens")
        from custom_models import PrefixModel
        model = PrefixModel(model, prefix_length=prefix_prompt_config['num_prepend_tokens'])
        # Ensure classifier remains trainable after wrapping
        for name, param in model.named_parameters():
            if 'classifier' in name: param.requires_grad = True

    elif finetune_mode == 'soft-prompt':
        print(f"   - Using Soft-prompt Tuning with {prefix_prompt_config['num_prepend_tokens']} tokens")
        from custom_models import SoftPromptModel
        model = SoftPromptModel(model, prompt_length=prefix_prompt_config['num_prepend_tokens'])
        # Ensure classifier remains trainable after wrapping
        for name, param in model.named_parameters():
            if 'classifier' in name: param.requires_grad = True

    elif finetune_mode == 'soft-prompt+lora':
        print(f"   - Using LoRA (r={lora_config['r']}, alpha={lora_config['alpha']}) + Soft-prompt ({prefix_prompt_config['num_prepend_tokens']} tokens)")
        # 1. Transform model into LoRA format
        lora_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=lora_config['r'], lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'], bias="none", target_modules=["query", "key", "value"]
        )
        model = get_peft_model(model, lora_peft_config)
        # 2. Wrap with SoftPromptModel
        from custom_models import SoftPromptModel
        model = SoftPromptModel(model, prompt_length=prefix_prompt_config['num_prepend_tokens'])
        # 3. Set trainable parameters: LoRA + Soft Prompt + Classifier
        for name, param in model.named_parameters():
            if 'lora_' in name or 'soft_prompt' in name or 'classifier' in name:
                param.requires_grad = True
            # Freeze the copy of classifier generated by LoRA wrapper
            if 'original_module' in name:
                param.requires_grad = False

    elif finetune_mode == 'prefix+lora':
        print(f"   - Using Prefix Tuning ({prefix_prompt_config['num_prepend_tokens']} tokens) + LoRA (r={lora_config['r']}, alpha={lora_config['alpha']})")
        # 1. Transform model into LoRA format
        lora_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=lora_config['r'], lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'], bias="none", target_modules=["query", "key", "value"]
        )
        model = get_peft_model(model, lora_peft_config)
        # 2. Wrap with PrefixModel
        from custom_models import PrefixModel
        model = PrefixModel(model, prefix_length=prefix_prompt_config['num_prepend_tokens'])
        # 3. Set trainable parameters: LoRA + Prefix + Classifier
        for name, param in model.named_parameters():
            if 'lora_' in name or 'prefix_modules' in name or 'classifier' in name:
                param.requires_grad = True
            # Freeze the copy of classifier generated by LoRA wrapper
            if 'original_module' in name:
                param.requires_grad = False

    elif finetune_mode == 'ia3':
        print(f"   - Using IA3")
        from custom_models import wrap_bert_for_ia3
        model = wrap_bert_for_ia3(model)
        # Ensure classifier and IA3 modules are trainable
        for name, param in model.named_parameters():
            if 'classifier' in name or 'ia3' in name:
                param.requires_grad = True

    else:
        raise ValueError(f"Unknown fine-tuning mode: {finetune_mode}.")

    # Calculate and print number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(f"   Trainable parameters: {trainable_param_count} ({trainable_param_count/total_params:.2%} of total)")
    if finetune_mode != 'full' and finetune_mode != 'last-layer':
        classifier_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n)
        peft_params_count = trainable_param_count - classifier_params_count
        print(f"     - Classifier parameters: {classifier_params_count} ({classifier_params_count/total_params:.2%})")
        print(f"     - Other parameters: {peft_params_count} ({peft_params_count/total_params:.2%})")

    return model, trainable_params


def run_training_loop(model, optimizer, train_dataloader, eval_dataloader, device, config, output_dir, tokenizer):
    epochs = config['epochs']
    enable_dp = config['enable_dp']
    
    best_accuracy = 0.0
    train_metrics = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"   Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            optimizer.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits
            
            if not enable_dp:
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
            else:
                loss = F.cross_entropy(logits, labels, reduction="none")
                optimizer.step(loss=loss) 
                batch_loss = loss.mean().item()
            
            total_loss += batch_loss
            num_batches += 1
            progress_bar.set_postfix({"loss": batch_loss})

        # Evaluate after each epoch
        avg_loss = total_loss / num_batches
        eval_results = evaluate_model(model, eval_dataloader, device)
        accuracy = eval_results['accuracy']
        print(f"   Epoch {epoch+1}/{epochs} - train loss: {avg_loss:.4f}, val acc: {accuracy:.4f}", end="")
        
        train_metrics.append({ "epoch": epoch + 1, "train_loss": avg_loss, "eval_accuracy": accuracy })
        
        # Save best model based on validation accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f" (New best accuracy!)", end="")
            model_path = os.path.join(output_dir, "best_model")
            if hasattr(model, 'save_pretrained'):
                 model.save_pretrained(model_path)
            else: # Fallback for non-PEFT wrapped models (e.g., finetune)
                 torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
                 model.config.save_pretrained(model_path) # Save config too
            tokenizer.save_pretrained(model_path)
        print()

    return best_accuracy, train_metrics


def train_model_on_dataset(dataset_name, config, tokenizer):
    print(f">> Training on {dataset_name} dataset..." )
    print(f"   Config: Mode={config['finetune_mode']}, LR={config['learning_rate']}, BS={config['batch_size']}, Epochs={config['epochs']}")
    if config['enable_dp']:
        print(f"   DP Enabled: ε={config['dp_config']['target_epsilon']}, MaxGradNorm={config['dp_config']['max_grad_norm']}")
    else:
        print(f"   DP Disabled")

    # Determine number of labels
    num_labels = 3 if dataset_name == "mnli" else 2
    
    # Prepare Model
    model, trainable_params = prepare_model(config, num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare Data
    train_dataloader, eval_dataloader, train_sample_size = prepare_data(config, dataset_name, tokenizer)

    # Prepare Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config['learning_rate'], weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999)
    )

    # Setup output directory
    timestamp = datetime.now().strftime('%y%m%d.%H%M%S')
    dp_tag = f"dp_eps{config['dp_config']['target_epsilon']}_" if config['enable_dp'] else ""
    output_dir = f"./logs/{dataset_name}_{config['finetune_mode']}_{dp_tag}{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Saving logs and models to: {output_dir}")

    # Initialize privacy engine if DP is enabled
    privacy_engine = None
    if config['enable_dp']:
        import warnings
        warnings.filterwarnings("ignore", message="Using a non-full backward hook.*") # Suppress specific warning

        # Monkey-patch private_transformers if needed
        if type(model) not in pe_module.SUPPORTED_TRANSFORMERS:
            pe_module.SUPPORTED_TRANSFORMERS += (type(model),)
            print(f"   Patched private_transformers to support model type: {type(model)}")

        privacy_engine = pe_module.PrivacyEngine(
            model,
            batch_size=config['batch_size'] * (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            sample_size=train_sample_size,
            epochs=config['epochs'],
            max_grad_norm=config['dp_config']['max_grad_norm'],
            target_epsilon=config['dp_config']['target_epsilon'],
        )
        privacy_engine.attach(optimizer)
        print(f"   Privacy Engine attached.")

    # Run Training Loop
    best_accuracy, train_metrics = run_training_loop(
        model, optimizer, train_dataloader, eval_dataloader, device, config, output_dir, tokenizer
    )

    # Save final model state
    final_model_path = os.path.join(output_dir, "final_model")
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(final_model_path)
    else:
        os.makedirs(final_model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(final_model_path, "pytorch_model.bin"))
        model.config.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"   Final model saved to {final_model_path}")

    # Final evaluation using the final model state
    model.eval()
    final_eval_results = evaluate_model(model, eval_dataloader, device)

    # Print results for this dataset
    print(f"\n===== {dataset_name.upper()} RESULTS =====")
    print(f"Best validation Acc: {best_accuracy:.4f}")
    print(f"Final validation Acc: {final_eval_results['accuracy']:.4f}")
    print("=" * (len(dataset_name) + 20)) # Adjust bar length
    print()

    final_eval_results["best_accuracy"] = best_accuracy
        
    return final_eval_results


if __name__ == "__main__":
    config = setup_config()

    tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])

    # Train on all specified datasets
    results = {}
    for dataset_name in config['datasets_to_train']:
        results[dataset_name] = train_model_on_dataset(dataset_name, config, tokenizer)

    # Print summary of results
    print("===== SUMMARY OF RESULTS =====")
    print(f"Model: {config['model_checkpoint']}")
    print(f"Fine-tuning mode: {config['finetune_mode']}")
    if 'lora' in config['finetune_mode']:
        print(f"   LoRA config: r={config['lora_config']['r']}, alpha={config['lora_config']['alpha']}, dropout={config['lora_config']['dropout']}")
    if 'prefix' in config['finetune_mode'] or 'soft-prompt' in config['finetune_mode']:
        print(f"   Prefix/Soft-prompt config: num_prepend_tokens={config['prefix_prompt_config']['num_prepend_tokens']}")
    print(f"Learning rate: {config['learning_rate']}, Batch size: {config['batch_size']}, Epochs: {config['epochs']}")
    print(f"Differential Privacy: {'Enabled' if config['enable_dp'] else 'Disabled'}")
    if config['enable_dp']:
        print(f"   ε={config['dp_config']['target_epsilon']}, MaxGradNorm={config['dp_config']['max_grad_norm']}")
    print("-" * 30)
    for dataset_name, result in results.items():
        print(f"{dataset_name}: Best Acc={result.get('best_accuracy')}")
    print("=" * 30)
