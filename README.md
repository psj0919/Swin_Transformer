# Swin Transformer

## Overview
This project integrates the Swin Transformer as the backbone in a semantic segmentation pipeline, replacing traditional CNN-based architectures such as ResNet. Swin Transformer is a hierarchical vision Transformer using shifted windows for self-attention, allowing for efficient and scalable modeling.

## Motivation
Unlike ViT which suffers from high computational costs due to global self-attention (`O(N^2)` complexity), Swin Transformer introduces:
- **Local Window-based attention** with linear complexity (`O(N)`),
- **Hierarchical representation** via patch merging,
- **Shifted window mechanism** for cross-window interaction.

This results in improved performance and efficiency for dense prediction tasks such as segmentation.

---

##  Architecture
The segmentation model follows this structure:
```
Swin Transformer Backbone -> UPerNet Decoder -> Semantic Segmentation Output
```

###  Swin Transformer Stages:
1. **Stage 1**: Linear embedding of 4�4 patches.
2. **Stage 2~4**: Patch merging layers progressively reduce resolution while increasing feature depth.
3. **Shifted Windows**: Enable global context modeling via local windows and alternating shifts.

### UPerNet Head:
- Pyramid Pooling Module (PPM)
- Feature Pyramid Network (FPN) for multi-scale feature fusion
