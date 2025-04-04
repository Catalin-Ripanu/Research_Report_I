# Quantum Federated Learning Framework for Generative Models

## Overview
This research investigates the integration of quantum computing with machine learning, specifically focusing on federated generative learning techniques in the NISQ (Noisy Intermediate-Scale Quantum) era. The project explores combining quantum computing approaches with generative adversarial networks (GANs) and denoising diffusion models (DDMs) through federated learning frameworks.

## Research Goals
- Assess quantum computing viability for machine learning in the NISQ era
- Apply quantum logic to federated generative learning techniques
- Develop quantum-enhanced GANs and DDMs using federated approaches
- Generate synthetic data across various datasets within computational constraints
- Explore knowledge distillation for adapting QNNs across different qubit network architectures

## Current Progress
The research has completed initial binary classification experiments using:
- Quantum Federated Learning (QFL)
- Quantum Convolutional Neural Networks (QCNNs)

### Experimental Setup
- **Dataset**: Synthetic hierarchical dataset with excited/non-excited cluster-state quantum data
- **Frameworks**: TensorFlow Quantum and TensorFlow Federated
- **QCNN Architecture**: 
  - Three pairs of quantum convolution-pooling layers
  - Quantum fully connected layer
  - Final measurement layer
  - 8 qubits with 64 qurons per setup

### Performance Evaluation
- **Server Optimizer**: SGD (learning rate = 1)
- **Client Optimizers**: Adam, RMSprop, SGD with varying learning rates
- **Validation**: Performed on subset of clients (restricted access environment)

## Related Work
- QDDMs generating samples using quark and gluon jet data from LHC Compact Muon Solenoid (CMS) [1]
- QCNN classification capabilities in medical domains [2, 3]
- Quantum data for distributed learning in quantum computing networks [4]
- Contextual resources [5]

## Next Steps
- Apply the QFL framework to develop generative adversarial networks with QCNNs
- Produce initial synthetic samples
- Test on diverse datasets and client configurations
- Further evaluate limitations and benefits of the framework and quantum models
- Implement knowledge distillation in federated learning for heterogeneous local model architectures

## Technologies
- TensorFlow Quantum [6]
- TensorFlow Federated [7]
- Quantum Neural Networks (QNNs)
- Quantum Convolutional Neural Networks (QCNNs)

## Computational Constraints
Current experiments limited to 8 qubits due to computational resource restrictions

# Setup Instructions for Running Experiments

To run the quantum federated learning experiments using the provided environment.yml file, follow these step-by-step instructions:

## Prerequisites
- Anaconda or Miniconda installed on your system
- Git (to clone the repository if needed)
- Sufficient computational resources for quantum simulations

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Catalin-Ripanu/Research_Report_I.git
   cd Research_Report_I
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate qfl-env  # The actual name will be specified in your environment.yml
   ```

3. **Verify installation**:
   ```bash
   python -c "import tensorflow as tf; import tensorflow_quantum as tfq; print(f'TensorFlow version: {tf.__version__}, TensorFlow Quantum version: {tfq.__version__}')"
   ```

## Running the Experiments

1. **Run evaluation**:
   ```bash
   # Example: Adjust experiments directory
   python3 qfl_paper.py --experiment_dir my_experiments
   ```

## Troubleshooting

If you encounter environment issues:
```bash
conda clean --all
conda env remove -n qfl-env  # Use your actual environment name
conda env create -f environment.yml
```

For CUDA/GPU compatibility issues:
```bash
# Check if TensorFlow sees your GPU
python -c "import tensorflow as tf; print('GPUs available:', tf.config.list_physical_devices('GPU'))"
```

## Credits

[1]: Mariia Baidachna, Rey Guadarrama, Gopal Ramesh Dahale, Tom Magorsch, Isabel
Pedraza, Konstantin T Matchev, Katia Matcheva, Kyoungchul Kong, and Sergei
Gleyzer. Quantum diffusion model for quark and gluon jet generation. arXiv preprint
arXiv:2412.21082, 2024.

[2]: Seunghyeok Oh, Jaeho Choi, and Joongheon Kim. A tutorial on quantum convolutional
neural networks (qcnn). In 2020 International Conference on Information and Commu-
nication Technology Convergence (ICTC), pages 236–239. IEEE, 2020.

[3]: Taieba Tasnim, Mohammad Rahman, and Fan Wu. Comparison of cnn and qcnn perfor-
mance in binary classification of breast cancer histopathological images. In 2024 IEEE
International Conference on Big Data (BigData), pages 3780–3787. IEEE, 2024.

[4]: Michael A Nielsen. Cluster-state quantum computation. Reports on Mathematical
Physics, 57(1):147–161, 2006

[5]: Mahdi Chehimi and Walid Saad. Quantum federated learning with quantum data. In
ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Pro-
cessing (ICASSP), pages 8617–8621. IEEE, 2022

[6]: Michael Broughton, Guillaume Verdon, Trevor McCourt, Antonio J Martinez, Jae Hyeon
Yoo, Sergei V Isakov, Philip Massey, Ramin Halavati, Murphy Yuezhen Niu, Alexan-
der Zlokapa, et al. Tensorflow quantum: A software framework for quantum machine
learning. arXiv preprint arXiv:2003.02989, 2020.

[7]: Ziteng Sun, Peter Kairouz, Ananda Theertha Suresh, and H Brendan McMahan. Can
you really backdoor federated learning? arXiv preprint arXiv:1911.07963, 2019
