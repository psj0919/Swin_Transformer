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
<img width="654" height="199" alt="image" src="https://github.com/user-attachments/assets/6e55fe59-e3cd-4531-a35e-14ec247b0d84" />

###  Swin Transformer Stages:
1. **Stage 1**: Linear embedding of 4X4 patches.
2. **Stage 2~4**: Patch merging layers progressively reduce resolution while increasing feature depth.
3. **Shifted Windows**: Enable global context modeling via local windows and alternating shifts.
<img width="577" height="156" alt="image" src="https://github.com/user-attachments/assets/72589673-6d97-461b-93e7-201991a81cd6" />

<img width="612" height="176" alt="image" src="https://github.com/user-attachments/assets/356d5a43-89dd-4669-a9a3-fa4d592aada3" />

### UPerNet Head:
- Pyramid Pooling Module (PPM)
- Feature Pyramid Network (FPN) for multi-scale feature fusion

  <img width="612" height="355" alt="image" src="https://github.com/user-attachments/assets/c5568292-549a-4a69-a659-d486b38b7e0d" />

