# IACLR: Intention Alignment via Contrastive Learning for Bipartite Graph Recommendation

## Requirements:
```
torch==2.1.0
dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html 
scipy
numpy
tensorboardX==1.8
pandas
tqdm
scikit-learn
sentence-transformers
transformers
```

## Training Command:
`python main.py --model lgn_intent --multi_view --dataset amazon`