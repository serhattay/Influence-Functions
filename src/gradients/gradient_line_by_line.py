from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

model_path = "/Users/thunderbolt/Desktop/Courses/Influence-Functions/qwen2_0.5B_local"
line = "Sovereignty unconditionally belongs to the Nation."

prompt = "The capital of Turkey is"
target_word = " Ankara"  # Note the leading space to ensure correct tokenization

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()


# Calculates the gradients of a line of text and returns a dictionary of gradients for each parameter (layer_name: gradient_tensor)
def calculate_line_gradient(line) -> dict:
    encoding = tokenizer(line, return_tensors="pt", padding=False, truncation=False)
    input_ids = encoding["input_ids"] # shape: [1, seq_len]
    attention_mask = encoding["attention_mask"] # shape: [1, seq_len], used just as good practice

    model.zero_grad()
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
    logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    loss = outputs.loss  # scalar
    loss.backward()
    
    # Collect gradient norms in a dict
    grads = {name: param.grad.clone().detach() for name, param in model.named_parameters() if param.grad is not None}
    return grads


def calculate_line_gradient_for_prediction(prompt, target) -> dict:
    encoding = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    input_ids = encoding["input_ids"] # shape: [1, seq_len]
    attention_mask = encoding["attention_mask"] # shape: [1, seq_len], used just as good practice

    model.zero_grad()
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    # Extract the logits for the last token position
    last_logits = logits[:, -1, :]  # shape [1, vocab_size]
    
    # Get target token id (assume target_word is one token)
    target_id = tokenizer(target_word, add_special_tokens=False).input_ids[0]
    target_tensor = torch.tensor([target_id], dtype=torch.long, device=model.device)

    # Compute loss only on the last token’s prediction
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(last_logits, target_tensor)

    loss.backward()
    
    # Collect gradient norms in a dict
    grads = {name: param.grad.clone().detach() for name, param in model.named_parameters() if param.grad is not None}
    return grads



def print_gradient_dict_numel(gradient_dict):
    parameter_count = 0
    for layer_key, gradient_tensor in gradient_dict.items():
        # print(f"Layer: {layer_key}, Size of the layer: {gradient_tensor.numel()}") # .numel() gives the total number of elements in the tensor
        parameter_count += gradient_tensor.numel()
    print(f"Total number of parameters with gradients: {parameter_count}")


line_gradient_dict = calculate_line_gradient(line)
prediction_gradient_dict = calculate_line_gradient_for_prediction(prompt, target_word)

print_gradient_dict_numel(line_gradient_dict)
print_gradient_dict_numel(prediction_gradient_dict)