# YOLOv8 Object Detection with Feature-Guided Distillation

An enhanced YOLOv8 object detection framework incorporating Feature-Guided Distillation (FGD) for improved model performance through knowledge distillation.

## Introduction

This project extends the Ultralytics YOLOv8 framework through a new feature oriented distillation (FGD) mechanism. In order to adapt to human-vehicle-pet object detection,the FGD method utilizes a multi-scale deformable attention converter (MSDATrans) to map features of certain layer in the model to teacher-style features, and guides the original model to learn teacher-style features at the same time.

## Features

- **Feature-Guided Distillation**: Advanced knowledge distillation using MSDATrans
- **Multi-Scale Attention**: Self-attention and cross-attention mechanisms across multiple scales
- **Flexible Configuration**: Support for various YOLOv8 model sizes and configurations

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.8
- CUDA >= 10.2 (optional but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision
pip install ultralytics
pip install numpy matplotlib opencv-python
```

## Model Architecture

### MSDATrans (Multi-Scale Deformable Attention Transformer)

The core component of FGD, MSDATrans consists of:

1. **Self-Attention**: Captures spatial dependencies within each feature map
2. **Cross-Attention**: Enables information exchange across different scales
3. **Feature Fusion**: Combines self-attended and cross-attended features using gating mechanisms
4. **Deformable Sampling**: Dynamically samples features based on learned offsets

### Distillation loss calculation process (Adversarial Training Perspective)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Adversarial Distillation Training Process               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Adversarial Game                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │     ┌─────────────────┐                        ┌─────────────────┐  │   │
│  │     │    MSDATrans    │◄────── Game ────────► │   Student Model │  │   │
│  │     │(Feature Transform)│                        │    (Learner)    │  │   │
│  │     └────────┬────────┘                        └────────┬────────┘  │   │
│  │              │                                          │           │   │
│  │              │ Generate "teacher-style" features        │ Learn to  │   │
│  │              │                                          │ match     │   │
│  │              ▼                                          ▼           │   │
│  │     f_trans (transformed)                        s_fs (student)      │   │
│  │              │                                          │           │   │
│  │              │              ┌─────────────┐             │           │   │
│  │              └─────────────►│ Adversarial  │◄────────────┘           │   │
│  │                            │    Loss      │                          │   │
│  │                            └──────┬──────┘                          │   │
│  │                                   │                                 │   │
│  │              ┌────────────────────┼────────────────────┐            │   │
│  │              ▼                    ▼                    ▼            │   │
│  │     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │   │
│  │     │  tea_loss   │    │ dist_loss   │    │  L_task    │          │   │
│  │     │ Update MSDATrans    │ Update Student │    │Detection Loss│          │   │
│  │     │ (every 5 batches)  │ (every batch) │    │ (every batch)│          │   │
│  │     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │   │
│  │            │                  │                  │                   │   │
│  │            ▼                  ▼                  ▼                   │   │
│  │     MSDATrans params    Student params    Student params            │   │
│  │     update (distill)    update (distill)   update (task)           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                     │
│                                      ▼                                     │
│                    Knowledge Transfer & Feature Alignment                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Adversarial Training Dynamics:**

| Player | Role | Objective | Update Frequency |
|--------|------|-----------|------------------|
| **MSDATrans** | Feature Transformer | Generate "teacher-style" features | Every 5 batches |
| **Student Model** | Learner | Match the transformed features | Every batch |

**Loss Formulas (Adversarial Perspective):**
- **Transformer Loss**: `tea_loss = sum(1 / (MSE(f_trans[j], s_fs[j]) + 1e-6))` — Encourage information preservation
- **Student Loss**: `dist_loss = mean(MSE(s_fs[i], f_trans[i].detach()))` — Match transformed features
- **Total Loss**: `L_total = L_task + dist_loss` — Combine detection task with distillation

**Key Insights:**
1. **Two-player Game**: MSDATrans and Student Model compete in an adversarial manner
2. **Information Flow**: Transformer generates features, Student learns to match
3. **Alternating Optimization**: Players take turns updating to find equilibrium
4. **Knowledge Distillation**: Student learns by matching the transformed features that mimic teacher style
```

## Configuration

### Model Configuration

Edit `ultralytics/cfg/models/v8/yolov8.yaml` to customize model architecture.

### Distillation Configuration

Modify parameters in `FGD.py`:

```python
msd_teacher = MSDATrans(
    base_channels=64,
    num_inputs=3,
    num_offsets=4
).to(device)

optimizer_msd = optim.Adam(
    msd_teacher.parameters(), 
    lr=1e-2, 
    betas=(0.9, 0.999)
)
```

### Training Configuration

```yaml
# Train settings
epochs: 200
batch: 8
imgsz: 640
lr0: 0.01
lrf: 0.01

# Distillation settings
distill: True
distill_ids: [17, 20, 23]
```

## Results

### Performance Comparison

| Model | Size | mAP@0.5 | mAP@0.5:0.95 | FPS |
|-------|------|---------|--------------|-----|
| YOLOv8n | 6.2M | 37.3 | 19.5 | 140 |
| YOLOv8n (with FGD) | 6.2M | 40.1 | 21.3 | 135 |
| YOLOv8l | 68.2M | 50.1 | 28.8 | 45 |

### Training Curves

The FGD approach demonstrates:
- Faster convergence compared to standard training
- Better generalization with less overfitting
- Improved performance on small objects

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525)


## Contact

For questions or support, please open an issue or contact the maintainers.

---

*Built with ❤️ using Ultralytics YOLOv8*