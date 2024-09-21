# Machine Learning and Ising Model for Finance

This project applies machine learning techniques to study financial systems through the lens of the Ising model. It leverages synthetic data generated from the Ising model and uses a Multi-Layer Perceptron (MLP) classifier to explore classification tasks. The project consists of data generation, modeling, and evaluation, with comprehensive explanations found in the notebooks and Python scripts.

## Project Structure

- **`data_generator.py`**: A Python script that generates synthetic data based on the Ising model for use in machine learning tasks. This script includes:
  - Functions for generating random temperatures within certain regimes.
  - Parallel processing to efficiently handle large datasets.
  - Data is categorized into distinct temperature regimes around the critical temperature `Tc`, based on predefined thresholds.
  
- **`Data_generator.ipynb`**: A Jupyter notebook that provides an in-depth explanation of how the data is generated using the Ising model. This notebook covers:
  - Theoretical background on the Ising model and its application to financial systems.
  - Step-by-step data generation process, including regime categorization and handling.
  - Visualization of the generated data to ensure correctness.
  
- **`My_MLP_Classifier.ipynb`**: The core machine learning notebook that trains a custom MLP classifier to classify the generated data into temperature regimes. It contains:
  - A detailed walkthrough of the MLP architecture.
  - Code for training, evaluating, and analyzing the performance of the classifier.
  - Discussion of hyperparameter tuning and the challenges of interpretability in financial machine learning models.
  
- **`ising4finance.py`**: A script implementing the Ising model in a financial context. Key functionalities include:
  - A Metropolis-Hastings algorithm for simulating the 2D Ising model.
  - Free energy updates, spin configurations, and field strength adjustments.
  - Optimization functions to minimize the energy state of the system.

## Getting Started

### Prerequisites

- Python 3.11
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `multiprocessing`
  - `h5py`
  
### Running the Code

1. **Generate Data**:
   - Run the `data_generator.py` script to generate synthetic data.
   - You can adjust the number of samples and regimes in the script as per your requirements.

   Example:
   ```bash
   python data_generator.py

### Train the MLP Classifier:

- Open the `My_MLP_Classifier.ipynb` notebook to train the MLP on the generated data.
- Follow the step-by-step explanations to understand how the data is processed, the MLP is trained, and the results are analyzed.

### Run Ising Model Simulations:

- The `ising4finance.py` script can be used to simulate the Ising model with adjustable parameters such as temperature, array size, and magnetic field strength.

## Project Insights

- The Ising model, commonly used in statistical mechanics, is applied here to financial systems to generate complex datasets. By simulating different regimes, we mimic market conditions under varying levels of stress or volatility.
- The MLP classifier effectively distinguishes between these regimes, demonstrating the power of machine learning in uncovering hidden structures within financial data.
- The combination of physics-based modeling (Ising model) and machine learning offers a novel perspective on financial data analysis.
