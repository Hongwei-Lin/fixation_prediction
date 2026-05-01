# Fixation Prediction on Medical Images

A deep learning project predicting where radiologists fixate when reading medical images (mammography, chest X-rays). The model learns saliency maps weighted by fixation dwell time, with the goal of producing attention predictions that align with how expert radiologists visually search images.

## Project goals

- Build a saliency prediction model for medical imaging using foundation-model backbones (DINOv2, etc.)
- Study cross-domain transfer from natural-image saliency (SALICON) → chest X-ray gaze data → mammography
- Investigate dwell-time-weighted ground truth construction as a more clinically meaningful supervision signal
- Validate on a unique dataset of digitized film mammograms with radiologist eye-tracking

## Status

Active development — environment setup complete, data pipeline in progress.

## Stack

- PyTorch (CUDA 12.1)
- Python 3.11
- W&B for experiment tracking

## License

