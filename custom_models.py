import torch
import torch.nn as nn
import os
import json


# ========================
# for IA3
# ========================
class IA3Scaling(nn.Module):
    def __init__(self, base_linear, device):
        super().__init__()
        self.base = base_linear
        self.ia3_alpha = nn.Parameter(torch.ones(base_linear.out_features, device=device), requires_grad=True)
        # Buffer to store intermediate output for gradient calculation
        self.register_buffer("intermediate_output", None, persistent=False)

    def forward(self, x):
        out_base = self.base(x)
        self.intermediate_output = out_base.detach().clone()
        out_final = out_base * self.ia3_alpha

        # Hook to compute and assign per-sample gradient for ia3_alpha
        def _capture_grad_sample(grad):
            # grad: dL/d(out_final) (shape: [batch_size, ..., alpha_index])

            # Calculate per-sample gradient: dL/d(alpha) = dL/d(out_final) * out_base
            # Sum over dims - all except first (batch_size) and last (alpha_index)
            dims_to_sum = tuple(range(1, grad.dim() - 1))
            per_sample_grad = (grad * self.intermediate_output).sum(dim=dims_to_sum)

            # Assign dL/d(alpha) per sample (shape: [batch_size, alpha_index])
            self.ia3_alpha.grad_sample = per_sample_grad.detach()
            return

        # Attach hook to the output tensor if requires_grad
        if out_final.requires_grad:
            out_final.register_hook(_capture_grad_sample)
        else:
             # If output doesn't require grad, ensure grad_sample is not present or cleared
             if hasattr(self.ia3_alpha, 'grad_sample'):
                 del self.ia3_alpha.grad_sample

        return out_final


def wrap_bert_for_ia3(model):
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if "attention.self.key" in name or "attention.self.value" in name or "intermediate.dense" in name:
            parent = get_parent_module(model, name)
            attr = name.split('.')[-1]
            original = getattr(parent, attr)
            setattr(parent, attr, IA3Scaling(original, device))
    return model


def get_parent_module(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent


# ========================
# for Soft Prompt
# ========================
class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, embedding_dim, device):
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim, device=device))

    def forward(self, batch_size):
        return self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
    
    
class SoftPromptModel(nn.Module):
    def __init__(self, base_model, prompt_length=10):
        super().__init__()
        self.base_model = base_model
        self.prompt_length = prompt_length

        embedding_dim = base_model.bert.embeddings.word_embeddings.embedding_dim
        self.soft_prompt_module = SoftPromptEmbedding(prompt_length, embedding_dim, base_model.device)

        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Make soft prompt trainable
        self.trainable_params = (
            list(self.soft_prompt_module.parameters()) +
            list(self.base_model.classifier.parameters())
        )

        # Set requires_grad for the selected trainable parameters
        for param in self.trainable_params:
            param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, **kwargs):

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            batch_size = input_ids.shape[0]
            input_embeds = self.base_model.bert.embeddings.word_embeddings(input_ids)
        else:
            batch_size = inputs_embeds.shape[0]
            input_embeds = inputs_embeds

        # 1. Expand soft prompts for batch
        expanded_prompt = self.soft_prompt_module.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Assign grad_sample directly to soft_prompt (required for DP optimization)
        def capture_grad_sample(grad):
            self.soft_prompt_module.soft_prompt.grad_sample = grad.detach().clone()
        expanded_prompt.register_hook(capture_grad_sample)

        # 3. Concatenate prompt and input embeddings
        final_inputs_embeds = torch.cat([expanded_prompt, input_embeds], dim=1)

        # 4. Extend attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.prompt_length, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask[:, :input_embeds.shape[1]]], dim=1)

        if token_type_ids is not None:
            prompt_token_types = torch.zeros(batch_size, self.prompt_length, dtype=token_type_ids.dtype, device=token_type_ids.device)
            token_type_ids = torch.cat([prompt_token_types, token_type_ids[:, :input_embeds.shape[1]]], dim=1)

        return self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=final_inputs_embeds,
            **kwargs
        )

    def save_pretrained(self, save_directory):
        """Saves the soft prompt and classifier state dictionary."""
        os.makedirs(save_directory, exist_ok=True)

        soft_prompt_path = os.path.join(save_directory, "soft_prompt.pt")
        torch.save(self.soft_prompt_module.state_dict(), soft_prompt_path)

        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.base_model.classifier.state_dict(), classifier_path)


# ========================
# For Prefix
# ========================
class PrefixEmbedding(nn.Module):
    def __init__(self, prefix_length, hidden_dim, device):
        super().__init__()
        self.prefix = nn.Parameter(torch.randn(prefix_length, hidden_dim, device=device))

    def forward(self, batch_size):
        expanded = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)

        def _capture_grad_sample(grad):
            self.prefix.grad_sample = grad.detach()

        expanded.register_hook(_capture_grad_sample)
        return expanded

class PrefixModel(nn.Module):
    def __init__(self, base_model, prefix_length=10):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.encoder_layers = base_model.bert.encoder.layer
        self.num_layers = len(self.encoder_layers)

        # Create separate prefix for each layer
        self.prefix_modules = nn.ModuleList([
            PrefixEmbedding(prefix_length, base_model.config.hidden_size, base_model.device)
            for _ in range(self.num_layers)
        ])

        self._register_prefix_hooks()

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Enable training for prefix_modules and classifier
        self.trainable_params = list(self.prefix_modules.parameters()) + list(self.base_model.classifier.parameters())
        for param in self.trainable_params:
            param.requires_grad = True

    def _register_prefix_hooks(self):
        def prepend_prefix_hook(layer_id):
            def hook(module, input):
                hidden_states = input[0]
                batch_size = hidden_states.size(0)
                prefix = self.prefix_modules[layer_id](batch_size)
                new_hidden_states = torch.cat([prefix, hidden_states], dim=1)
                return (new_hidden_states,)
            return hook

        for i, layer in enumerate(self.encoder_layers):
            layer.register_forward_pre_hook(prepend_prefix_hook(i))

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        for i, prefix in enumerate(self.prefix_modules):
            torch.save(prefix.state_dict(), os.path.join(save_directory, f"prefix_layer_{i}.pt"))

        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.base_model.classifier.state_dict(), classifier_path)
