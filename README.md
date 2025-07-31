# VAE Drift Detection System

A concept drift detection system for data streams using Variational Autoencoders (VAE). This project was developed as part of a Master's thesis at Poznań University of Technology.

## 📖 Description

This project implements a novel approach to concept drift detection in machine learning from data streams, utilizing Variational Autoencoders (VAE). The system offers an alternative to traditional drift detection methods, providing better effectiveness in detecting subtle changes in complex, multi-dimensional data.

## 🎯 Project Goals

- Development of a VAE-based concept drift detector
- Effective detection of changes in streaming data distribution
- Maintaining acceptable computational performance
- Providing better precision-recall trade-off than existing methods
- Integration with popular incremental learning libraries (River)

## 🏗️ Project Structure

```
├── models/                          # Drift detection models
│   ├── vae_model.py                 # Variational Autoencoder implementation
│   ├── batch_vae_drift_detector.py  # Batch processing drift detector
│   ├── vae_drift_detector.py        # Basic VAE drift detector
│   └── vae_drift_adaptive_predictor.py # Adaptive predictor
├── stream_groups/                   # Stream group definitions
│   ├── composition_basic.txt        # Basic composition drifts
│   ├── composition_complex.txt      # Complex composition drifts
│   ├── imbalance_basic.txt         # Basic imbalance drifts
│   └── ...                         # Other stream groups
├── new_streams/                     # ARFF datasets
├── results/                         # Experiment results
├── plots/                          # Charts and visualizations
├── experiment.py                   # Main experiment script
├── utilities.py                    # Helper functions
├── final_results_explorer.ipynb    # Results analysis notebook
└── environment.yml                 # Conda environment definition
```

## 🚀 Installation

### Requirements

- Python 3.10+
- Conda (recommended)

### Environment Setup

1. Clone the repository:
```bash
git clone <https://github.com/jakubgolab/drift-detection-system.git>
cd <drift-detection-system>
```

2. Create Conda environment:
```bash
conda env create -f environment.yml
conda activate ml_env
```

3. Verify installation:
```bash
python -c "import torch, river, pandas; print('Installation completed successfully!')"
```

## 📊 Datasets

The project uses synthetic data streams in ARFF format that simulate various types of concept drifts. The datasets are generated using the **imbalanced-stream-generator** framework.

### Drift Types:
- **Locality drifts** (Move, Join) - changes in class positions in feature space
- **Composition drifts** (Split) - splitting existing classes into new ones
- **Imbalance drifts** (Im, Rare, Borderline) - changes in class proportions

### Stream Categories:
- `composition_basic` - basic composition drifts
- `locality_basic` - basic locality drifts
- `imbalance_basic` - basic imbalance problems
- `*_complex` - complex combinations of different drift types

## 🔧 Usage

### Running Experiments

```bash
python experiment.py <stream_group_name>
```

Example:
```bash
python experiment.py composition_basic
```

### Available Stream Groups:
- `composition_basic`
- `composition_complex`
- `composition_with_imbalance`
- `locality_basic`
- `locality_with_imbalance`
- `imbalance_basic`
- `imbalance_combinations`
- `imbalance_complex`
- `imbalance_example_types`

### Stream Categorization

```bash
python stream-categorizer.py
```

## 🧠 Models

### VAE Drift Detector

Main detector utilizing Variational Autoencoders:

```python
from models.batch_vae_drift_detector import BatchVAEDriftDetector

detector = BatchVAEDriftDetector(
    input_dim=5,          # Number of data dimensions
    hidden_dim=32,        # Hidden layer size
    latent_dim=2,         # Latent space dimension
    p_threshold=0.05,     # p-value threshold
    epochs=50,            # Training epochs
    window_size=5000,     # Window size
    retrain_on_drift=True # Whether to retrain after drift
)
```

### Compared Detectors

The system compares VAE with traditional methods:
- **ADWIN** - Adaptive Windowing
- **KSWIN** - Kolmogorov-Smirnov Windowing
- **Page-Hinkley** - Page-Hinkley Test

## 📈 Evaluation Metrics

The system evaluates drift detection performance using confusion matrix metrics and detection timing analysis:

### Classification Metrics
- **True Positives (TP)** - correctly detected drift occurrences
- **False Positives (FP)** - incorrectly detected drifts (false alarms)
- **True Negatives (TN)** - correctly identified stable periods 
- **False Negatives (FN)** - missed drift occurrences

### Detection Timing Metrics
- **Detection Delays** - time lag between actual drift occurrence and detection (measured in windows)

## 📊 Results

Experiment results are saved in the `results/` folder and include:
- `metrics.pkl` - evaluation metrics including TP, FP, TN, FN, accuracy, precision, recall, F1-score, G-mean, and detection timing
- `results.pkl` - detailed detection results with detection timestamps and reconstruction errors
- `drift_detection.log` - experiment logs
- Charts visualizing drift detection performance across different dataset groups

### Analysis Categories
The results are organized by dataset groups for comprehensive analysis:
- **Splits** - composition drift scenarios (class splitting)
- **Joins** - locality drift scenarios (class merging)  
- **Moves** - locality drift scenarios (class movement)
- **Borderlines** - borderline imbalance scenarios
- **Imbalances** - various imbalance ratio scenarios

## 📓 Notebooks

### `final_results_explorer.ipynb`
Comprehensive analysis notebook for drift detection results:

**Features:**
- **Dataset Group Selection** - filter analysis by drift type (Splits, Joins, Moves, Borderlines, Imbalances)
- **Performance Metrics Visualization** - line plots and subplots for TP, FP, FN across models and datasets
- **Model Comparison** - side-by-side comparison of all drift detectors
- **Detection Delay Analysis** - visualization of detection timing performance
- **Summary Statistics** - mean performance metrics across dataset groups
- **Automated Plot Generation** - saves publication-ready PDF charts

**Available Visualizations:**
- True/False Positives/Negatives trends per dataset
- Overall model comparison across metrics
- Detection delay analysis
- Individual model performance breakdown

## 🔬 Methodology

### VAE Architecture

1. **Encoder**: Transforms input data to distribution parameters in latent space (μ, log σ²)
2. **Reparameterization**: Enables differentiation during training
3. **Decoder**: Reconstructs data from latent space samples
4. **Loss Function**: Combines reconstruction error and KL divergence

### Detection Process

1. Train VAE on reference window
2. Calculate reconstruction errors for new data
3. Statistical test (Kolmogorov-Smirnov) comparing error distributions
4. Detect drift when p-value < threshold

## ⚙️ Configuration

### VAE Detector Parameters:
- `input_dim`: Input data dimension
- `hidden_dim`: Hidden layer size (default: 32)
- `latent_dim`: Latent space dimension (default: 2)
- `p_threshold`: p-value threshold (default: 0.05)
- `window_size`: Window size (default: 5000)
- `epochs`: Training epochs (default: 50)
- `retrain_on_drift`: Whether to retrain after drift

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Error**: If you don't have GPU, VAE automatically uses CPU
2. **Memory Issues**: Reduce `batch_size` or `window_size`
3. **Slow Training**: Consider increasing `learning_rate` or reducing `epochs`

### Logs:
Check `.log` files in the `results/` folder for detailed error information.

## 📚 References

This project is based on cutting-edge research in:
- Variational Autoencoders (VAE)
- Concept Drift Detection
- Stream Learning
- Imbalanced Data Streams

## 📚 Acknowledgments

### Special Thanks

We would like to express our special gratitude to the authors of the **imbalanced-stream-generator** framework:

- **Dariusz Brzezinski**
- **Leandro L. Minku**
- **Tomasz Pewinski**
- **Jerzy Stefanowski**
- **Artur Szumaczuk**

for developing and maintaining the *imbalanced-stream-generator* framework, which provided the synthetic datasets used in this research.

The *imbalanced-stream-generator* has been instrumental in generating diverse and realistic imbalanced data streams with various types of concept drifts and data difficulty factors, enabling comprehensive evaluation of our VAE-based drift detection approach.

**Reference Paper:**  
Brzezinski, D., Minku, L. L., Pewinski, T., Stefanowski, J., & Szumaczuk, A. (2021). *The impact of data difficulty factors on classification of imbalanced and concept drifting data streams*. Knowledge and Information Systems, 63, 1429–1469. [https://doi.org/10.1007/s10115-021-01560-w](https://doi.org/10.1007/s10115-021-01560-w)

**Framework Repository:**  
[https://github.com/dabrze/imbalanced-stream-generator](https://github.com/dabrze/imbalanced-stream-generator)


## 👥 Author

- Jakub Gołąb </br>

Project developed as part of a Master's thesis at Poznań University of Technology.

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome:
- Bug reports (Issues)
- Feature proposals (Pull Requests)
- Methodology discussions

---

**Note**: This project is under development. Some functionalities may be experimental.
