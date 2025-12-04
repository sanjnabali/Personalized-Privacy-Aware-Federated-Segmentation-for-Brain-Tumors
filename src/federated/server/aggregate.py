import torch
import os

def aggregate_models(model_paths, save_path):
    """
    Performs Federated Averaging (FedAvg):
    AVG = (Model_1 + Model_2 + ... + Model_N) / N
    """
    if not model_paths:
        print("[Server] No models to aggregate!")
        return None
        
    print(f"\n[Server] Aggregating {len(model_paths)} client models...")
    
    # 1. Load the first model to initialize the sum
    # map_location='cpu' prevents OOM if aggregating on GPU
    avg_state = torch.load(model_paths[0], map_location="cpu")
    
    # 2. Sum up the parameters from the rest
    for i in range(1, len(model_paths)):
        state = torch.load(model_paths[i], map_location="cpu")
        for key in avg_state:
            # We assume all models have the exact same architecture keys
            avg_state[key] += state[key]
            
    # 3. Divide by number of clients to get the Mean
    num_clients = len(model_paths)
    for key in avg_state:
        # Check if the parameter is a floating point tensor (weights/bias)
        if torch.is_floating_point(avg_state[key]):
            avg_state[key] = avg_state[key] / num_clients
        else:
            # For integer tensors (like global_step), we just take the first one or int div
            avg_state[key] = avg_state[key] // num_clients
        
    # 4. Save the New Global Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(avg_state, save_path)
    print(f"[Server] New Global Model Saved: {save_path}")
    
    return save_path