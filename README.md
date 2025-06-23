# MAML-Transformer-IDS
Framework for few-shot learning in network intrusion detection
This user manual provides a comprehensive guide to the Meta_Learning_V1.ipynb code, which implements Model-Agnostic Meta-Learning (MAML) with a Transformer architecture for network intrusion detection.
1. Introduction
This code provides a complete framework for few-shot learning in network intrusion detection. It leverages a Transformer architecture within the MAML framework to enable models to quickly adapt to new, unseen attack types with limited examples.
2. Setup and Installation
Prerequisites:
•	Python 3.x
Recommended Installation: Install the necessary libraries using pip:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

GPU Configuration (Optional): The code includes logic to configure TensorFlow for GPU usage, which is recommended for faster training. Ensure you have compatible GPU drivers and the CUDA toolkit installed.
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
3. Core Components
The codebase is organized into several classes, each handling a specific aspect of the meta-learning pipeline:
•	NetworkDataProcessor: Manages dataset loading, preprocessing, and the creation of meta-learning tasks (support and query sets). It supports UNSW-NB15, NSL-KDD, and CICIDS2017 datasets, and can generate synthetic data for demonstration purposes.
•	TransformerBlock: Implements a standard Transformer encoder block, including Multi-Head Attention and a Feed-Forward Network.
•	PositionalEncoding: Adds positional information to input embeddings, crucial for the Transformer to understand sequence order.
•	ExpandDimsLayer: A custom Keras layer used to expand dimensions of input tensors, necessary for the Transformer's input format.
•	MAMLTransformer: Encapsulates the MAML algorithm using a Transformer-based neural network as its base model. It handles inner-loop adaptation and outer-loop meta-updates.
•	MAMLTrainer: Orchestrates the entire meta-training process, including training loops, validation, early stopping, and logging.
•	SecurityAnalyzer: Provides utilities for evaluating the model's performance from a security perspective, offering classification reports, confusion matrices, ROC curves, and attention weight visualizations.
4. Usage Examples
Here’s a typical workflow for using the code:
4.1. Data Loading and Preprocessing
# Initialize data processor (if data_path is None, synthetic data is generated)
data_path = None # Set to your dataset path, e.g., 'data/UNSW_NB15_dataset.csv'
data_processor = NetworkDataProcessor(data_path=data_path)

# Load data (synthetic data is used if data_path is None)
df = data_processor.load_data()

# Preprocess data
X, y_binary, y_multiclass = data_processor.preprocess_data(df)
print("Attack mapping:", data_processor.attack_mapping)
4.2. Creating Meta-Learning Tasks
n_way = 5      # Number of classes per task
k_shot = 5     # Number of support examples per class
query_size = 15 # Number of query examples per class
num_meta_train_tasks = 800
num_meta_val_tasks = 100
num_meta_test_tasks = 100

train_tasks = data_processor.create_tasks(X, y_multiclass, num_tasks=num_meta_train_tasks, k_shot=k_shot, query_size=query_size)
val_tasks = data_processor.create_tasks(X, y_multiclass, num_tasks=num_meta_val_tasks, k_shot=k_shot, query_size=query_size)
test_tasks = data_processor.create_tasks(X, y_multiclass, num_tasks=num_meta_test_tasks, k_shot=k_shot, query_size=query_size)

print(f"Generated {len(train_tasks)} training tasks, {len(val_tasks)} validation tasks, {len(test_tasks)} test tasks.")
print(f"Each task is {n_way}-way {k_shot}-shot.")
4.3. Initializing and Training the MAML Model
# Initialize MAML Transformer model
maml_model = MAMLTransformer(
    input_shape=(X.shape[1],),
    n_way=n_way,
    k_shot=k_shot,
    inner_lr=0.01,
    meta_lr=0.001,
    meta_batch_size=16,
    num_inner_updates=5,
    embed_dim=256,
    num_heads=8,
    ff_dim=512,
    num_transformer_blocks=4,
    mlp_units=[128, 64],
    dropout=0.1
)

# Initialize MAML Trainer
trainer = MAMLTrainer(
    maml_model=maml_model,
    train_tasks=train_tasks,
    val_tasks=val_tasks,
    test_tasks=test_tasks,
    meta_epochs=1000,
    meta_batch_size=16,
    eval_interval=100,
    early_stopping_patience=10,
    log_dir='logs'
)

# Train the model
history = trainer.train()

# Visualize training history
trainer.visualize_training()

4.4. Evaluating the MAML Model
import os

# Load the best meta-model weights for evaluation (if training was stopped)
maml_model.load_meta_model(os.path.join('logs', 'best_model.weights.h5'))

# Evaluate on test tasks
test_accuracy, test_loss = maml_model.evaluate(test_tasks)
print(f"Final Test Accuracy: {test_accuracy:.4f}, Final Test Loss: {test_loss:.4f}")

# Plot adaptation curve for a sample test task
sample_test_task = test_tasks[0]
trainer.plot_adaptation_curve(sample_test_task)
4.5. Analyzing Attack Types
import numpy as np

# Initialize Security Analyzer
security_analyzer = SecurityAnalyzer(
    maml_model=maml_model,
    data_processor=data_processor,
    attack_mapping=data_processor.attack_mapping,
    log_dir='logs'
)

# Evaluate on test tasks for full metrics
y_true_all, y_pred_all, y_scores_all_binary = security_analyzer.evaluate_model_on_tasks(test_tasks)

# Generate and save classification report
security_analyzer.generate_classification_report(y_true_all, y_pred_all)

# Plot and save confusion matrix
security_analyzer.plot_confusion_matrix(y_true_all, y_pred_all)

# Plot and save ROC curve
security_analyzer.plot_roc_curve(y_true_all, y_scores_all_binary)

# Analyze performance across specific attack types
attack_tasks = []
attack_names = []
for attack_name in ['dos', 'probe']: # Extend with other attack types as needed
    if attack_name in data_processor.attack_mapping:
        cls_id = data_processor.attack_mapping[attack_name]
        attack_X = X[y_multiclass == cls_id]
        attack_y = np.full(len(attack_X), 0) # Placeholder, actual labels handled by create_tasks

        if len(attack_X) >= k_shot + query_size:
            task = data_processor.create_tasks(attack_X, attack_y, num_tasks=1, k_shot=k_shot, query_size=query_size)[0]
            attack_tasks.append(task)
            attack_names.append(f"Attack: {attack_name}")

if attack_tasks:
    performance = security_analyzer.analyze_attack_types(attack_tasks, attack_names)
    print("\nPerformance across attack types:")
    for name, acc, f1 in zip(performance['attack_names'], performance['accuracies'], performance['f1_scores']):
        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")
.6. Visualizing Attention Weights
# Visualize attention weights for explainability
# Use your preprocessed X and corresponding labels (multiclass or binary)
security_analyzer.visualize_attention_weights(X, y_multiclass if y_multiclass is not None else y_binary)
5. Reproducibility
The code ensures reproducibility by setting random seeds for NumPy, TensorFlow, and Python's random module.
6. Troubleshooting
•	ValueError: Multiclass labels are required for creating few-shot tasks: Ensure y_multiclass is provided to create_tasks and that your data (or synthetic data generation) includes multiclass labels.
•	GPU Issues: Verify correct installation of GPU drivers and CUDA toolkit compatible with your TensorFlow version. tf.config.experimental.set_memory_growth can help with memory allocation.
•	File Not Found: Confirm the data_path in NetworkDataProcessor is correct for custom datasets.
7. Customization and Advanced Usage
•	Dataset Integration: Modify NetworkDataProcessor.load_data to support new datasets.
•	Hyperparameter Tuning: Experiment with parameters like inner_lr, meta_lr, num_inner_updates, meta_batch_size, n_way, k_shot, and query_size in MAMLTransformer and MAMLTrainer to optimize performance.
•	Transformer Architecture: Adjust embed_dim, num_heads, ff_dim, num_transformer_blocks, mlp_units, and dropout within MAMLTransformer for model capacity control.
•	Targeted Attack Analysis: Create specific tasks for individual attack types to perform granular performance analysis.
•	Attention Visualization: Change the layer_name in security_analyzer.visualize_attention_weights to explore attention patterns in different Transformer blocks.


