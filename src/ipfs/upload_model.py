import requests
import os

def upload_file(file_path):
    """
    Uploads a file to the local IPFS node (port 5001) and returns the CID.
    """
    if not os.path.exists(file_path):
        print(f"[IPFS] File not found: {file_path}")
        return None

    try:
        # 'rb' mode is crucial for binary files like models
        files = {'file': open(file_path, 'rb')}
        
        # Connect to IPFS API 'add' endpoint
        response = requests.post('http://127.0.0.1:5001/api/v0/add', files=files)
        
        if response.status_code == 200:
            # IPFS returns JSON: {'Name': '...', 'Hash': 'Qm...'}
            data = response.json()
            cid = data['Hash']
            print(f"  ☁️ [IPFS] Uploaded {os.path.basename(file_path)} -> CID: {cid}")
            return cid
        else:
            print(f"[IPFS] Upload Failed: {response.text}")
            return None
    except Exception as e:
        print(f"[IPFS] Connection Error: {e}")
        print("   Is IPFS Desktop running?")
        return None