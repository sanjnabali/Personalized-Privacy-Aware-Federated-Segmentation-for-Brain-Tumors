import os
import json
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LEDGER_PATH = os.path.join(PROJECT_ROOT, "experiments", "blockchain_ledger.json")

def load_ledger():
    if not os.path.exists(LEDGER_PATH):
        # Create a dummy ledger for demonstration if real one doesn't exist yet
        print(f"⚠️ Ledger not found at {LEDGER_PATH}. Generating MOCK data for visualization...")
        return generate_mock_ledger()
    
    with open(LEDGER_PATH, "r") as f:
        return json.load(f)

def generate_mock_ledger():
    """Generates fake blockchain data to test the visualizer"""
    return [
        {
            "index": 0,
            "timestamp": datetime.now().timestamp(),
            "event": "GENESIS",
            "global_model_cid": "QmGenesisHash000000000000000000000000000",
            "round_loss": 1.0
        },
        {
            "index": 1,
            "timestamp": datetime.now().timestamp() + 3600,
            "event": "ROUND_COMPLETE",
            "global_model_cid": "QmHashRound1xB7s8d9f0g1h2j3k4l5m6n7o8p9",
            "contributors": ["Client_1", "Client_3", "Client_4"],
            "round_loss": 0.85
        },
        {
            "index": 2,
            "timestamp": datetime.now().timestamp() + 7200,
            "event": "ROUND_COMPLETE",
            "global_model_cid": "QmHashRound2xC3d4e5f6g7h8i9j0k1l2m3n4o5",
            "contributors": ["Client_1", "Client_2", "Client_4"],
            "round_loss": 0.72
        },
        {
            "index": 3,
            "timestamp": datetime.now().timestamp() + 10800,
            "event": "ROUND_COMPLETE",
            "global_model_cid": "QmHashRound3xD5e6f7g8h9i0j1k2l3m4n5o6p7",
            "contributors": ["Client_2", "Client_3"],
            "round_loss": 0.61
        }
    ]

def visualize_chain_graph(ledger):
    """Draws the Blockchain as a directed graph"""
    G = nx.DiGraph()
    
    losses = []
    rounds = []
    
    print("\n🔗 __BLOCKCHAIN STATE__ 🔗")
    print(f"{'Block':<6} | {'CID (Hash)':<20} | {'Loss':<6} | {'Contributors'}")
    print("-" * 65)

    for i, block in enumerate(ledger):
        # 1. Console Output
        short_cid = block['global_model_cid'][:12] + "..."
        loss = block.get('round_loss', 0)
        contribs = str(len(block.get('contributors', []))) + " Clients" if 'contributors' in block else "N/A"
        
        print(f"#{block['index']:<5} | {short_cid:<20} | {loss:<6.4f} | {contribs}")
        
        # 2. Graph Node
        label = f"Block {block['index']}\nLoss: {loss:.3f}"
        G.add_node(block['index'], label=label)
        
        if i > 0:
            G.add_edge(i-1, i)
            
        losses.append(loss)
        rounds.append(block['index'])

    # 3. Plotting
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: The Chain Visual
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=42)
    # Arrange nodes linearly
    pos = {i: (i, 0) for i in rounds}
    
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", 
            font_size=10, font_weight="bold", arrowsize=20)
    plt.title("Immutable Ledger Chain (Trust Layer)")
    
    # Subplot 2: The Training Progress (Loss recorded on-chain)
    plt.subplot(1, 2, 2)
    plt.plot(rounds, losses, marker='o', linestyle='-', color='orange', linewidth=2)
    plt.title("On-Chain Validation Loss")
    plt.xlabel("Block Height (Round)")
    plt.ylabel("Dice Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ledger_data = load_ledger()
    visualize_chain_graph(ledger_data)