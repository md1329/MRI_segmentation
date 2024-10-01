brain_mri_segmentation/
│
├── models/           # To store your trained model weights
│   └── best_model.h5 # Pre-trained model (nested U-Net or attention U-Net)
├── app.py            # FAST API backend
├── frontend.py       # Streamlit frontend
└── utils.py          # Utility functions for preprocessing and postprocessing
