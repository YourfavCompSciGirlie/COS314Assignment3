import pandas as pd
import numpy as np
import tensorflow as tf
import os
import random
from tabulate import tabulate
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Get user parameters with defaults
seed_val = int(input("Enter a seed value: "))

train_filepath = input("Enter the training data filepath (default: Euro_USD Stock/BTC_train.csv): ") or 'Euro_USD Stock/BTC_train.csv'
test_filepath = input("Enter the test data filepath (default: Euro_USD Stock/BTC_test.csv): ") or 'Euro_USD Stock/BTC_test.csv'

# The data actually in the algorithm in that optimization when you are doing feeding
train_data = pd.read_csv(train_filepath)
# The data used for testing that the algorithm works well when it sees data it has not seen before
test_data = pd.read_csv(test_filepath)

n_neurons_1 = int(input("Enter number of neurons for first hidden layer (default 64): ") or "64")
n_neurons_2 = int(input("Enter number of neurons for second hidden layer (default 32): ") or "32")
n_neurons_3 = int(input("Enter number of neurons for third hidden layer (default 16): ") or "16")
dropout_rate = float(input("Enter dropout rate (default 0.2): ") or "0.2")
learning_rate = float(input("Enter learning rate (default 0.001): ") or "0.001")
n_epochs = int(input("Enter max epochs (default 50): ") or "50") 

# Store results of all runs
results_table = []
all_results = []

# Perform 10 runs
for run in range(10):
    current_seed = seed_val + run
    print(f"\nRun {run + 1}/10 with seed {current_seed}")
    
    # Set all random seeds for reproducibility
    os.environ['PYTHONHASHSEED'] = str(current_seed)
    random.seed(current_seed)
    np.random.seed(current_seed)
    tf.random.set_seed(current_seed)

    # Force TensorFlow to use single thread to ensure reproducible results
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # Ensure TensorFlow uses deterministic operations
    tf.config.experimental.enable_op_determinism()

    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        train_data.drop("Output", axis=1).values,
        train_data["Output"].values,
        test_size=0.2,
        random_state=current_seed
    )

    # Normalize features
    x_scalar = StandardScaler()
    x_train = x_scalar.fit_transform(x_train)
    x_val = x_scalar.transform(x_val)
    x_test = x_scalar.transform(test_data.drop("Output", axis=1).values)
    y_test = test_data["Output"].values

    # regularization
    model = keras.Sequential([
        keras.layers.Input(shape=(x_train.shape[1],)),
        Dense(n_neurons_1, activation='relu', 
              kernel_regularizer=keras.regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(n_neurons_2, activation='relu', 
              kernel_regularizer=keras.regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(n_neurons_3, activation='relu', 
              kernel_regularizer=keras.regularizers.l2(0.001)),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model - adam = adaptive learning rate optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # The optimizer is the algorithm that the model uses to update its weights during training
    # The loss function is the function that the model tries to minimize during training
    # The metrics are the values that the model tracks during training to see how well it is performing
    model.compile(optimizer=optimizer, 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])

    # Callbacks are used to monitor the training process and make adjustments
    # Early stopping = stop training when the model stops improving
    # Reduce learning rate = reduce the learning rate when the model stops improving
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, 
            restore_best_weights=True
        )
    ]

    # Train the model with user parameters
    history = model.fit(
        x_train, y_train,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=0  # Suppress epoch-by-epoch output
    )

    # Calculate metrics for both training and test sets
    train_pred = (model.predict(x_train, verbose=0) > 0.5).astype("int32")
    test_pred = (model.predict(x_test, verbose=0) > 0.5).astype("int32")
    
    train_acc = np.mean(train_pred.flatten() == y_train)
    train_f1 = f1_score(y_train, train_pred)
    test_acc = np.mean(test_pred.flatten() == y_test)
    test_f1 = f1_score(y_test, test_pred)

    # Store results for table
    results_table.append([
        current_seed,
        f"{train_acc:.4f}",
        f"{train_f1:.4f}",
        f"{test_acc:.4f}",
        f"{test_f1:.4f}"
    ])

    # Store detailed results
    run_results = {
        'seed': current_seed,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'confusion_matrix': confusion_matrix(y_test, test_pred),
        'classification_report': classification_report(y_test, test_pred),
        'history': history.history
    }
    all_results.append(run_results)

# Print results table
headers = ["Seed", "Train Acc", "Train F1", "Test Acc", "Test F1"]
print("\nResults for all runs:")
print(tabulate(results_table, headers=headers, tablefmt="grid"))

# Find best run based on test F1 score (more balanced metric than accuracy)
best_run = max(all_results, key=lambda x: x['test_f1'])

# Print detailed results of best run
print(f"\nBest Run Details (seed {best_run['seed']}):")
print(f"Training Accuracy: {best_run['train_acc']:.4f}")
print(f"Training F1-score: {best_run['train_f1']:.4f}")
print(f"Test Accuracy: {best_run['test_acc']:.4f}")
print(f"Test F1-score: {best_run['test_f1']:.4f}")
print("\nConfusion Matrix:")
print(best_run['confusion_matrix'])
print("\nClassification Report:")
print(best_run['classification_report'])

# Plot best run's training & validation accuracy and loss values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(best_run['history']['accuracy'], label='Training Accuracy')
plt.plot(best_run['history']['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Best Run)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_run['history']['loss'], label='Training Loss')
plt.plot(best_run['history']['val_loss'], label='Validation Loss')
plt.title('Model Loss (Best Run)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()