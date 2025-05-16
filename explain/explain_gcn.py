import os
import numpy as np
import tensorflow as tf
import argparse
import sys

# Allow imports from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import EllipticDatasetLoader
from models import GCNSequential
from models.layers import GCNLayer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Argument parser setup
parser = argparse.ArgumentParser(description="Explain GCN model on Elliptic dataset")

parser.add_argument("--datadir_path", type=str, default="elliptic_bitcoin_dataset")
parser.add_argument("--filter_unknown", type=str2bool, default=False)
parser.add_argument("--local_features_only", type=str2bool, default=True)
parser.add_argument("--use_scaler", type=str2bool, default=True)
parser.add_argument("--use_anova", type=str2bool, default=False)
parser.add_argument("--n_anova_features", type=int, default=50)
parser.add_argument("--use_pca", type=str2bool, default=True)
parser.add_argument("--n_pca_components", type=int, default=30)
parser.add_argument("--use_svd", type=str2bool, default=False)
parser.add_argument("--n_svd_components", type=int, default=30)
parser.add_argument("--weights_path", type=str, default="outputs/gcn_model_weights.h5")

args = parser.parse_args()

print("🔍 Loading dataset...")
dl = EllipticDatasetLoader(
    datadir_path=args.datadir_path,
    test_portion=0.3,
    filter_unknown=args.filter_unknown,
    local_features_only=args.local_features_only,
    use_scaler=args.use_scaler,
    use_anova=args.use_anova,
    n_anova_features=args.n_anova_features,
    use_pca=args.use_pca,
    n_pca_components=args.n_pca_components,
    use_svd=args.use_svd,
    n_svd_components=args.n_svd_components
)

print("🧠 Building model...")
model = GCNSequential([
    GCNLayer(64, "relu"),
    tf.keras.layers.Dropout(0.3),
    GCNLayer(dl.num_classes)
])

# Initialize model weights by calling it once
print("⚙️ Initializing model for weight loading...")
for _, dummy_x, _, dummy_adj in dl.train_batch_iterator():
    _ = model([dummy_adj, dummy_x], training=False)
    break  # Just one pass is enough

print("📂 Loading weights from:", args.weights_path)
model.load_weights(args.weights_path)

print("✅ Model ready. Starting explanation...")

# ===== PLACEHOLDER for your explainability logic =====
# For example, you can implement:
# - Gradient-based saliency
# - Integrated Gradients
# - Custom attention visualization

# Example: use one test batch to compute predictions
for _, x, y, adj in dl.test_batch_iterator():
    logits = model([adj, x], training=False)
    preds = tf.argmax(logits, axis=-1)
    print("Sample predictions:", preds.numpy()[:10])
    break

# ======================================================

print("✅ Explanation script finished.")
