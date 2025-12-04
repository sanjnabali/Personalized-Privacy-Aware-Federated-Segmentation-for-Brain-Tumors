import torch
import os
import sys
import time
import shutil # Needed to copy files
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------

from src.utils.config_loader import load_config
from src.federated.client.train_local import train_client
from src.federated.server.aggregate import aggregate_models
from src.ipfs.upload_model import upload_file

def deploy_contract(config):
    # ... (Same Blockchain deployment logic as before, abbreviated for brevity) ...
    # Assuming this part works fine from previous successful run.
    print("\n🔗 [Blockchain] Deploying Smart Contract...")
    rpc_url = config['blockchain']['rpc_url']
    private_key = config['blockchain']['private_key']
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected(): raise ConnectionError("Failed to connect to Ganache.")
    account = w3.eth.account.from_key(private_key)
    w3.eth.default_account = account.address
    try:
        install_solc('0.8.0'); set_solc_version('0.8.0')
    except: pass
    
    contract_path = os.path.join(project_root, "src/blockchain/smart_contracts/FederatedConsensus.sol")
    if os.path.exists(contract_path):
        with open(contract_path, "r") as f: contract_source = f.read()
    else:
        contract_source = "pragma solidity ^0.8.0; contract FederatedConsensus { constructor() {} function submitUpdate(string memory c, string memory i) public {} }"

    compiled_sol = compile_source(contract_source, output_values=['abi', 'bin'])
    contract_id, contract_interface = next(iter(compiled_sol.items()))
    Contract = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
    construct_txn = Contract.constructor().build_transaction({'from': account.address, 'nonce': w3.eth.get_transaction_count(account.address), 'gas': 2000000, 'gasPrice': w3.to_wei('20', 'gwei')})
    signed = w3.eth.account.sign_transaction(construct_txn, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"[Blockchain] Contract Deployed at: {tx_receipt.contractAddress}")
    return w3, w3.eth.contract(address=tx_receipt.contractAddress, abi=contract_interface['abi'])

def run_federated_learning():
    config = load_config()
    rounds = config['federated']['rounds']
    num_clients = config['federated']['num_clients']
    
    # 1. Setup Blockchain
    try:
        w3, contract = deploy_contract(config)
    except Exception as e:
        print(f"Blockchain Error: {e}")
        contract = None

    # 2. Init Global Model
    global_model_path = config['paths']['global_model']
    # Define the "Best" model path
    global_model_best_path = global_model_path.replace(".pt", "_best.pt")
    
    mae_weights = os.path.join(project_root, "saved_models/maes/mae_pretrained.pt")
    if os.path.exists(mae_weights):
        print(f"\n[Init] Seeding with MAE Weights.")
        model_state = torch.load(mae_weights)
        os.makedirs(os.path.dirname(global_model_path), exist_ok=True)
        torch.save(model_state, global_model_path)
    else:
        print(f"\n[Init] Random Initialization.")

    # 3. Training Loop
    start_time = time.time()
    best_loss = float('inf') # Track the best loss seen so far
    
    for r in range(1, rounds + 1):
        print(f"\n{'='*10} ROUND {r}/{rounds} {'='*10}")
        round_client_updates = []
        round_losses = []
        
        # A. Client Phase
        for c in range(1, num_clients + 1):
            # Capture LOSS returned by client
            cid, local_path, client_loss = train_client(c, global_model_path, r)
            
            round_client_updates.append(local_path)
            round_losses.append(client_loss)
            
            # Blockchain Log
            if contract:
                try:
                    print(f"   ⛓️ [Blockchain] Logging Client {c}...")
                    txn = contract.functions.submitUpdate(f"Client{c}", cid).build_transaction({
                        'from': w3.eth.default_account, 'nonce': w3.eth.get_transaction_count(w3.eth.default_account),
                        'gas': 500000, 'gasPrice': w3.to_wei('20', 'gwei')
                    })
                    signed_txn = w3.eth.account.sign_transaction(txn, config['blockchain']['private_key'])
                    w3.eth.send_raw_transaction(signed_txn.raw_transaction)
                except: pass

        # B. Aggregation Phase
        print("\n[Server] Aggregating...")
        new_global_path = aggregate_models(round_client_updates, global_model_path)
        
        # C. CHECKPOINTING: Is this the best model?
        avg_round_loss = sum(round_losses) / len(round_losses)
        print(f"[Stats] Round {r} Average Loss: {avg_round_loss:.4f}")
        
        if avg_round_loss < best_loss:
            print(f"⭐ [New Record] Loss improved ({best_loss:.4f} -> {avg_round_loss:.4f}). Saving BEST model.")
            best_loss = avg_round_loss
            # Copy the current global model to 'global_model_best.pt'
            shutil.copy(global_model_path, global_model_best_path)
        else:
            print(f"   [Info] Loss did not improve (Best: {best_loss:.4f}).")

        # Upload Global
        upload_file(new_global_path)

    print(f"\nFederated Learning Complete.")
    print(f"BEST Model saved at: {global_model_best_path} (Loss: {best_loss:.4f})")
    print(f"LAST Model saved at: {global_model_path}")

if __name__ == "__main__":
    run_federated_learning()