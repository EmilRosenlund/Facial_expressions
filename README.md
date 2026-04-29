# Facial Expression Recognition

A facial expression classifier trained on FER2013 dataset. Uses VGGFace2 backbone.

## Usage

Test on a small test set:
```bash
python test.py
```

Run live webcam inference:
```bash
python project/vggFace2/webcam_inference.py
```

## Dataset

- Training: 91,869 images
- Validation: 22,967 images
- Classes: angry, disgust, fear, happy, neutral, sad, surprise

## Model

- Best validation loss: 0.9677 (epoch 183)
- Pre-trained weights in `/best_model_v5.pth`