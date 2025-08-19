# Semantic Compression with VQ-VAE

A text compression system that uses Vector Quantized Variational Autoencoders (VQ-VAE) and FAISS indexing to compress sentence embeddings whilst preserving semantic meaning. Compressed representations can be reconstructed using similarity matching or Large Language Models.

## What it does

1. **Compresses text**: Converts sentences to SBERT embeddings, then compresses them using VQ-VAE quantisation
2. **Indexes efficiently**: Uses FAISS OPQ+PQ indexing for fast similarity search
3. **Reconstructs meaning**: Decodes compressed representations back to readable text via similarity matching or LLM reconstruction

## Requirements

- Python 3.8+
- CUDA compatible GPU recommended (but CPU works)
- API key for at least one LLM service (OpenAI, Anthropic, or Google) - optional for similarity-only mode

## Installation

```bash
git clone https://github.com/your-username/semantic-compression-vqvae.git
cd semantic-compression-vqvae
pip install -r requirements.txt
```

## Quick Start

### 1. Configure the system

Create `vqvae_config.json`:

```json
{
  "device": "cpu",
  "dataset_split": "train",
  "sample_size": 10000,
  "batch_size": 64,
  "test_fraction": 0.1,
  "pq_M": 8,
  "vqvae_hidden_dim": 128,
  "vqvae_embedding_dim": 64,
  "vqvae_num_embeddings": 512,
  "vqvae_learning_rate": 1e-3,
  "vqvae_epochs": 10,
  "embedding_dimension": 384
}
```

### 2. Train and compress

```bash
python pipeline.py
```

This loads Wikitext-2 data, generates SBERT embeddings, trains the VQ-VAE model, and saves compressed representations.

### 3. Decode/reconstruct

```bash
python decode.py
```

Reconstructs text from compressed representations using similarity matching (always works) or LLM reconstruction (requires API keys).

### 4. Evaluate performance

```bash
python evaluate.py
```

Generates metrics comparing reconstructed text to originals using cosine similarity, BERTScore, and BLEU.

## LLM API Configuration

Set environment variables for any LLM service you want to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export GOOGLE_API_KEY="your-google-key"
export SEMANTIC_RECON_BACKEND="openai"  # or "claude" or "gemini"
```

The system works without LLM APIs by using similarity matching for reconstruction.

## Project Structure

```
semantic-compression-vqvae/
├── pipeline.py          # Main training and compression pipeline
├── decode.py            # Reconstruction from compressed representations
├── evaluate.py          # Performance evaluation and metrics
├── llm_callers.py      # LLM API wrappers
├── requirements.txt     # Python dependencies
├── vqvae_config.json   # Configuration file (create this)
├── .gitignore
├── LICENSE
├── ACKNOWLEDGMENTS.md
└── artifacts/          # Generated models and data (created by pipeline.py)
```

## Technical Details

- **Embeddings**: Uses SBERT `all-MiniLM-L6-v2` model (384 dimensions)
- **Compression**: VQ-VAE with configurable codebook size and hidden dimensions
- **Indexing**: FAISS OPQ+PQ for efficient similarity search
- **Reconstruction**: Cosine similarity matching or LLM-based generation
- **Evaluation**: Multiple metrics including cosine similarity, BERTScore, BLEU

## Limitations

- Requires substantial training time for large datasets
- LLM reconstruction depends on external API availability
- Compression ratio varies based on dataset and model configuration
- CPU training is significantly slower than GPU

## Performance

Typical results on Wikitext-2 (10,000 samples):
- Compression ratios: 20-50x reduction in storage
- Reconstruction quality: >0.8 cosine similarity with originals
- Training time: 5-15 minutes (varies by hardware)

## Licensing

This software is available for:
- **Academic and research use** - Free under the included license
- **Personal experimentation** - Free for individual learning
- **Commercial use** - Requires paid licensing agreement

For commercial licensing: c42meitheal@gmail.com

## Contact

c42meitheal@gmail.com

## Contributing

Issues and pull requests welcome. See CONTRIBUTING.md for guidelines.

Note: Contributors retain rights to their contributions but grant permission for inclusion in both free and commercial versions of the software.
