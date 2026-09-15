# OMR — Optical Music Recognition

A from-scratch Optical Music Recognition (OMR) system that converts images of
classical sheet music into machine-readable symbolic music notation.

The long-term goal is to build an interactive web platform where a user can
upload sheet music and receive its corresponding musical notes or tablature.

## Project Status

🚧 **Currently in development**

The current implementation focuses on building the dataset pipeline and a
from-scratch CNN + Transformer baseline.

### Current progress

- [x] GrandStaff dataset integration
- [x] Dataset indexing and image/label pairing
- [x] Custom symbolic music tokenizer
- [x] Vocabulary construction
- [x] Dataset exploratory analysis
- [x] PyTorch Dataset implementation
- [x] Variable-width image batching
- [x] CUDA-enabled PyTorch environment
- [x] CNN visual encoder
- [ ] CNN → Transformer feature projection
- [ ] Transformer decoder
- [ ] Model training pipeline
- [ ] Model evaluation
- [ ] Inference pipeline
- [ ] FastAPI backend
- [ ] Interactive web interface
- [ ] Notes/tab conversion

---

## Overview

The system follows an image-to-sequence architecture:

```text
Sheet Music Image
       │
       ▼
   CNN Encoder
       │
       ▼
Visual Feature Sequence
       │
       ▼
Transformer Decoder
       │
       ▼
Symbolic Music Tokens
       │
       ▼
Notes / Tabs
