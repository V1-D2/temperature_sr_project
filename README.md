
# Temperature Super-Resolution with SwinIR + Real-ESRGAN

Hybrid model for 8x super-resolution of AMSR-2 temperature data.

## Setup
1. Install dependencies: \`pip install -r requirements.txt\`
2. Place NPZ files in \`data/\` directory
3. Run training: \`python train_temperature_sr.py --data_dir ./data --output_dir ./experiments\`

## Project Structure
- \`models/\`: Network architectures
- \`realesrgan/\`: Real-ESRGAN components
- \`data_preprocessing.py\`: Temperature data preprocessing
- \`hybrid_model.py\`: Combined SwinIR + Real-ESRGAN model
- \`train_temperature_sr.py\`: Main training script
- \`test_temperature_sr.py\`: Testing script
EOF