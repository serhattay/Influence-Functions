from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import numpy as np
from pprint import pprint

model_path = "/Users/thunderbolt/Desktop/Courses/Influence-Functions/qwen2_0.5B_local"
file_path = "/Users/thunderbolt/Desktop/Courses/Influence-Functions/src/wikipedia_api/wikipedia_article.txt"

prompt = "Dum vita est, spes est."

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Calculates the gradients for the next prediction token by token for the entire text
# Returns a dictionary of gradients for each token position (1-based index) in the text
# Each entry contains:
#   - textual_representation: str
#   - gradients: torch.Tensor (1D) or None for the first token
#   - error_while_predicting: float (loss value) or -1 for the first token
def token_by_token_gradient(text) -> dict:
    encoding = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
    input_ids = encoding["input_ids"] # shape: [1, seq_len]
    attention_mask = encoding["attention_mask"] # shape: [1, seq_len], used just as good practice

    # Clears previous gradients
    model.zero_grad()

    with torch.enable_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]
    
    # TODO: PROBLEM REGARDING THE LOGIT SHIFTING, ALSO CHECK THE LOSS FN CALCULATIONS LATER 
    shifted_logits = logits[:, :-1, :].contiguous()  # shape: [1, seq_len-1, vocab_size]
    shifted_targets = input_ids[:, 1:].contiguous()  # shape: [1, seq_len-1]
    
    logits_without_batch = shifted_logits.squeeze(0)  # shape: [seq_len-1, vocab_size]
    targets_without_batch = shifted_targets.squeeze(0)  # shape: [seq_len-1]
    
    # print(f"Shifted logits shape (only 1 batch): {logits_without_batch.shape}")
    # print(f"Shifted targets shape (only 1 batch): {targets_without_batch.shape}")

    # Use CrossEntropyLoss but keep individual losses for each position.
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Calculate loss per token prediction; result is flattened.
    # Shape: [seq_len - 1]
    individual_token_losses = loss_fn(
        logits_without_batch,                         # Reshape logits for loss function
        targets_without_batch                         # Reshape targets for loss function
    )
    
    # Dictionary to hold the results: parameter name -> list of token gradients.
    token_gradients = dict()
    
    # Store gradient for the first token (there is no prediction for it, so gradients are None)
    token_gradients[1] = {
        "textual_representation": tokenizer.decode(input_ids[0][0]), # Type: str
        "gradients": None,      # Type: torch.Tensor (1D)
        "error_while_predicting": -1 # Type: float (-1 is to indicate not calculated)
        }

    num_of_predictions = logits_without_batch.size(0) # This is seq_len - 1

    for t in range(num_of_predictions):

        model.zero_grad()
        
        # Select the loss while predicting the t'th token; individual_token_losses shape: [batch_size, seq_len-1]
        # print(individual_token_losses)
        current_token_loss_scalar = individual_token_losses[t]
        
        grads_for_token_t_tuple = torch.autograd.grad(
            outputs=current_token_loss_scalar,
            inputs=list(model.parameters()),
            retain_graph=True
        )

        # print(f"Number of separate layers:", len(grads_for_token_t_tuple))
        
        one_d_tensors = []
        for grad_tensor in grads_for_token_t_tuple:
            one_d_tensors.append(grad_tensor.view(-1))  # Flatten each gradient tensor to 1-D

        grads_for_token_t = torch.cat(one_d_tensors, dim=0).flatten()  # Flatten into a single 1-D tensor
        # print(f"Flattened gradient tensors for guessing the token number {t+2}: {grads_for_token_t}")

        dict_item_per_token = {
            "textual_representation": tokenizer.decode(input_ids[0][t+1]), # Type: str
            "gradients": grads_for_token_t,      # Type: torch.Tensor (1D)
            "error_while_predicting": current_token_loss_scalar.item() # Type: float (-1 is to indicate not calculated)
            }

        # Store gradient for the prediction step 't' using dictionary key 'k = t+2'.
        # This key 'k' aligns with the 1-based index of the token being *predicted* in the original text.
        # Example: The gradient from the loss of predicting the 2nd token (step t=0) is stored under key 2.
        # This ensures the gradient responsible for predicting token k is associated with index k.
        token_gradients[t+2] = dict_item_per_token

    return token_gradients

# Prints gradients for each token separately
pprint(token_by_token_gradient(prompt))