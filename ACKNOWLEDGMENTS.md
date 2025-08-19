# Acknowledgments

This project builds upon excellent work from the research and open source communities. We gratefully acknowledge:

## Core Technologies

**Vector Quantized Variational Autoencoders (VQ-VAE)**
- Original paper: "Neural Discrete Representation Learning" by van den Oord et al. (2017)
- Foundational work enabling discrete latent representations

**Sentence-BERT (SBERT)**
- Reimers & Gurevych: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)
- `sentence-transformers` library and pre-trained models
- https://www.sbert.net/

**FAISS (Facebook AI Similarity Search)**
- Meta AI Research for fast similarity search and clustering
- Enables efficient similarity indexing at scale
- https://faiss.ai/

## Libraries and Dependencies

**PyTorch Ecosystem**
- PyTorch team for the deep learning framework
- HuggingFace for transformer models and datasets

**Evaluation Metrics**
- BERTScore authors: Zhang et al. (2019)
- NLTK contributors for BLEU score implementation
- scikit-learn for machine learning utilities

**Data**
- Wikitext-2 dataset from Merity et al. (2016)
- Available through HuggingFace datasets

## LLM Integration
- OpenAI for GPT models API
- Anthropic for Claude models API  
- Google for Gemini models API

## Research Context

This implementation combines several established techniques in a novel pipeline:
- SBERT embeddings for semantic representation
- VQ-VAE compression for discrete quantisation
- FAISS indexing for efficient retrieval
- LLM reconstruction for text generation

Whilst each component has been thoroughly researched, their specific combination in this configuration appears to be a unique contribution to the semantic compression literature.

## Community

Thank you to the broader AI/ML research community for publishing reproducible research and maintaining high-quality open source libraries that make projects like this possible.
