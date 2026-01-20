# backend/scripts/download_datasets.py

import pandas as pd
import requests
from pathlib import Path

def download_liar_dataset():
    """Download LIAR dataset"""
    urls = {
        'train': 'https://raw.githubusercontent.com/tfs4/liar_dataset/master/train.tsv',
        'test': 'https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv',
        'valid': 'https://raw.githubusercontent.com/tfs4/liar_dataset/master/valid.tsv'
    }
    
    data_dir = Path('../data/raw/liar')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for name, url in urls.items():
        print(f"Downloading {name} data...")
        df = pd.read_csv(url, sep='\t', header=None)
        df.to_csv(data_dir / f'{name}.csv', index=False)
        print(f"✓ {name} data saved")

def download_fakenewsnet():
    """Download FakeNewsNet dataset"""
    # Instructions to clone FakeNewsNet repository
    print("Clone FakeNewsNet repository:")
    print("git clone https://github.com/KaiDMML/FakeNewsNet.git ../data/raw/FakeNewsNet")

if __name__ == "__main__":
    download_liar_dataset()
    download_fakenewsnet()