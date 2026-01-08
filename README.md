README

# Install everything

## If you want to run in a virtual environment:
```
py -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

```
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.0 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```


# Run programme 
```
py sign_language_stgcn.py --cache_dir "stgcn_cache" --epochs 150 --imbalance_strategy "loss_weight"
```


# Run streamlit app
```
streamlit run app.py
```