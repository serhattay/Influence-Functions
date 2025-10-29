from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import numpy as np

model_path = "/Users/thunderbolt/Desktop/Courses/Influence-Functions/qwen2_0.5B_local"
file_path = "/Users/thunderbolt/Desktop/Courses/Influence-Functions/src/wikipedia_api/wikipedia_article.txt"

prompt = "The capital of Turkey is"
target_word = " Ankara"  # Note the leading space to ensure correct tokenization

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()


# Calculates the gradients of a line of text and returns a dictionary of gradients for each parameter (layer_name: gradient_tensor) - V1
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
    
    grad_list = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_list.append(param.grad.clone().detach().view(-1))  # flatten

    if grad_list:
        all_grads = torch.cat(grad_list, dim=0)  # a 1-D PyTorch tensor
    else:
        all_grads = torch.tensor([], device=model.device)
    
    return all_grads

# Calculates the gradients for the prediction given a prompt and target word (calculates the gradient only for the last prediction) - V1
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
    
    grad_list = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_list.append(param.grad.clone().detach().view(-1))  # flatten

    if grad_list:
        all_grads = torch.cat(grad_list, dim=0)  # a 1-D PyTorch tensor
    else:
        all_grads = torch.tensor([], device=model.device)
    return all_grads


# Sends two gradient dictionaries and computes the inner product between them - V1
def calculate_line_to_pred_inner_product(line_gradient_vector, prediction_gradient_vector) -> float:
    inner_product = torch.dot(line_gradient_vector, prediction_gradient_vector)  # Element-wise multiplication and sum
    return inner_product

# Reads the file line by line, calculates the gradient for each line, and computes the inner product with the prediction gradient - V1
def read_and_calculate_inner_prod_of_lines_vs_pred(file_path, prompt, target_word):
    prediction_gradient_vector = calculate_line_gradient_for_prediction(prompt, target_word)
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
        for idx, line in enumerate(lines, start=1):
            line_gradient_vector = calculate_line_gradient(line)
            
            print(f"Line {idx} to prediction inner product: {calculate_line_to_pred_inner_product(line_gradient_vector, prediction_gradient_vector)}")



read_and_calculate_inner_prod_of_lines_vs_pred(file_path, prompt, target_word)