from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

model_path = "/Users/thunderbolt/Desktop/Courses/Influence-Functions/qwen2_0.5B_local"

# Send to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path)

model.to(device, dtype=dtype)

model.eval()


prompt = "The capital of Turkey is"
target_word = tokenizer.encode(" Ankara")

# Forward pass and get outputs
model.zero_grad()

# Returns pytorch tensors
tokenized_inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

# Generate a single token continuation
outputs = model(tokenized_inputs)
last_token_logits = outputs.logits[:, -1, :]                # shape : [batch, vocab_size]

print(f"Logits shape: {last_token_logits.shape}")

next_token_id = torch.argmax(last_token_logits, dim=-1)     # shape : [batch]
print(tokenizer.decode(next_token_id))

# Compute loss with the most likely token as target
loss_fn = nn.CrossEntropyLoss()

target_id = tokenizer(" Ankara", add_special_tokens=False)["input_ids"][0]

# Make a batch of size 1
target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)

loss = loss_fn(last_token_logits, target_tensor)

print(f"Loss: {loss.item()}")


# Compute backward pass
loss.backward()

# Print gradients for all parameters
print("\nGradient information:")
print("-" * 50)

total_norm = 0.0
counter = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        # Computes the L2 norm of the aggregation of all gradients in that particular parameter tensor
        grad_norm = param.grad.data.norm(2).item()
        total_norm += grad_norm ** 2
        print(f"Layer: {name}")
        print(f"Gradient L2 norm: {grad_norm:.6f}")
        print("-" * 50)
        counter += 1
        
# Print total gradient norm
total_norm = total_norm ** 0.5 # Takes square root to get the total norm
print(f"\nTotal gradient L2 norm: {total_norm:.6f}")
print(f"Number of parameter tensors with aggregated gradients: {counter}")
