# Bitcoin Price Movement Prediction Using Multilayer Perceptron (MLP)

## Technical Implementation Details

### How to run
- Make sure you are in the correct directory using: 
```bash
pip install -r requirements.txt
```
- Install dependencies using `pip install -r requirements.txt`
- Run the script using: 
```bash
python mlp-model.py
```
- Note that the script takes a while to run since it performs 10 runs with different random seeds

### Neural Network Architecture

#### Layer Configuration
1. **Input Layer**
   - Dynamic shape adapting to feature count
   - Direct dense connections

2. **Hidden Layer Structure**
```python
Dense(64, activation='relu', kernel_regularizer=l2(0.001))
Dropout(0.2)
Dense(32, activation='relu', kernel_regularizer=l2(0.001))
Dropout(0.2)
Dense(16, activation='relu', kernel_regularizer=l2(0.001))
```
- First Hidden: 64 neurons + ReLU
- Second Hidden: 32 neurons + ReLU
- Third Hidden: 16 neurons + ReLU
- L2 Regularization: λ=0.001
- Dropout Rate: 20%

3. **Output Configuration**
```python
Dense(1, activation='sigmoid')
```
- Single neuron for outputting binary classification

### Training Implementation

#### Model Configuration
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy

#### Training Parameters
- Batch Size: 32 samples
- Maximum Epochs: 50
- Early Stopping: 5-epoch patience
- Validation Split: 20% of training data

### Performance Analysis

The model performs 10 runs with incrementing seeds, tracking:
- Training Accuracy
- Training F1-score
- Test Accuracy
- Test F1-score

Results are displayed in a table format:
```
+--------+-------------+------------+------------+-----------+
|   Seed |   Train Acc |   Train F1 |   Test Acc |   Test F1 |
+========+=============+============+============+===========+
|  XXXXX |     0.XXXX  |    0.XXXX  |     0.XXXX |    0.XXXX |
```

### Implementation Features

#### Reproducibility Measures
- Seed-based initialization
- Single-threaded operations
- Deterministic TensorFlow operations

#### Data Processing
- StandardScaler normalization
- Train/validation split (80/20)
- Binary classification output

#### Model Optimization
- L2 regularization (λ=0.001)
- Dropout layers (20%)
- Early stopping with best weights restoration

### Visualization

The script generates two plots for the best performing run:
1. **Model Accuracy Plot**
   - Training vs Validation accuracy
   - Epoch-wise progression

2. **Model Loss Plot**
   - Training vs Validation loss
   - Learning convergence visualization

### Best Run Analysis
For each execution, the script reports:
- Detailed metrics for best performing seed
- Confusion matrix
- Classification report with precision, recall, F1-score
- Support values for each class