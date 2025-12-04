import torch
import requests  # <--- Standard library, never breaks
from web3 import Web3
from solcx import install_solc
import os
from dotenv import load_dotenv

load_dotenv()

def check_setup():
    print("--- CHECKING SYSTEM SETUP (PYTHON 3.11.9) ---")
    
    # 1. Check GPU/PyTorch
    print(f"[1] PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"    GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("    WARNING: GPU not detected. Using CPU (Safe for small 100-image dataset).")

    # 2. Check Blockchain (Ganache)
    try:
        w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
        if w3.is_connected():
            print(f"[2] Blockchain: CONNECTED to Ganache (Chain ID: {w3.eth.chain_id})")
        else:
            print("[2] Blockchain: FAILED. Please ensure Ganache is open.")
    except Exception as e:
        print(f"[2] Blockchain Error: {e}")

    # 3. Check IPFS (Direct HTTP)
    try:
        # Tries to reach the IPFS 'ID' endpoint at port 5001
        response = requests.post('http://127.0.0.1:5001/api/v0/id')
        if response.status_code == 200:
            ipfs_id = response.json().get('ID', 'Unknown')
            print(f"[3] IPFS: CONNECTED (ID: {ipfs_id})")
        else:
            print(f"[3] IPFS Connection Failed: Status {response.status_code}")
    except Exception as e:
        print(f"[3] IPFS Error: {e}. Is IPFS Desktop running?")

    # 4. Check Solidity Compiler
    print("[4] Checking Solidity...")
    try:
        # This installs the compiler binary compatible with your OS
        install_solc('0.8.0')
        print("    Solidity 0.8.0 ready.")
    except Exception as e:
        print(f"    Solidity Error: {e}")

if __name__ == "__main__":
    check_setup()