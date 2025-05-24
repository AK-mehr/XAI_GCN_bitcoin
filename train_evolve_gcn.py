import numpy as np
import tensorflow as tf

from data_loader import EllipticDatasetLoader
from models import EvolveGCN
from models.layers import EGCUH, HGRUCell, SummarizeLayer, GCNLayer

# For f1, CM
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import argparse


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
parser = argparse.ArgumentParser(description="Train GCN on Elliptic dataset")

parser.add_argument("--datadir_path", type=str, default="elliptic_bitcoin_dataset")
parser.add_argument("--test_portion", type=float, default=0.3)
parser.add_argument("--filter_unknown", type=str2bool, default=False)
parser.add_argument("--local_features_only", type=str2bool, default=True)
parser.add_argument("--use_scaler", type=str2bool, default=False)
parser.add_argument("--use_anova", type=str2bool, default=False)
parser.add_argument("--n_anova_features", type=int, default=50)
parser.add_argument("--use_pca", type=str2bool, default=False)
parser.add_argument("--n_pca_components", type=int, default=30)
parser.add_argument("--use_svd", type=str2bool, default=False)
parser.add_argument("--n_svd_components", type=int, default=30)
parser.add_argument("--class_weights", nargs=3, type=float, default=[0.7, 0.29, 0.01])
parser.add_argument("--num_epoch", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=1e-3)

args = parser.parse_args()


DATADIR = "elliptic_bitcoin_dataset"
FILTER_UNKNOWN = False
ONLY_LOCAL_FEATURE = False
CLASS_WEIGTHS = [0.7,0.29,0.01]
NUM_ROLLS = 4
TEST_SHARE = 0.3
NUM_EPOCH = args.num_epoch
LEARNING_RATE = args.learning_rate


def reset_metrics(list_of_metrics):
    for m in list_of_metrics:
        m.reset_states()


print('==============================================')
print("Preparing the dataset...")
print('==============================================')

# dl = EllipticDatasetLoader(DATADIR, TEST_SHARE, FILTER_UNKNOWN,local_features_only=ONLY_LOCAL_FEATURE)
dl = EllipticDatasetLoader(
    datadir_path=DATADIR,
    test_portion=TEST_SHARE,
    filter_unknown=FILTER_UNKNOWN,
    local_features_only=ONLY_LOCAL_FEATURE,
    use_scaler=args.use_scaler,
    use_anova=args.use_anova,
    n_anova_features=args.n_anova_features,
    use_pca=args.use_pca,
    n_pca_components=args.n_pca_components,
    use_svd=args.use_svd,
    n_svd_components=args.n_svd_components
)

print('==============================================')
print('Build the model...')

model = EvolveGCN([
    EGCUH(HGRUCell(64),SummarizeLayer(),activation="relu"),
    EGCUH(HGRUCell(dl.num_classes),SummarizeLayer())
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def run_model(adj,nodes,targets,training=False):
    weigths = tf.reduce_sum(CLASS_WEIGTHS * targets, axis=-1)
    states = model.get_initial_weigths(tf.shape(nodes))
    logits = []
    for i in range(NUM_ROLLS):
        l, states = model([adj, nodes, states], training=training)
        logits.append(l)

    loss = sum(loss_func(targets, l, sample_weight=weigths) for l in logits)
    return logits[-1], loss, weigths


train_loss_metric = tf.keras.metrics.Mean()
train_accuracy_metric = tf.keras.metrics.Accuracy()
train_precision_metric = tf.keras.metrics.Precision()
train_recall_metric = tf.keras.metrics.Recall()
test_loss_metric = tf.keras.metrics.Mean()
test_accuracy_metric = tf.keras.metrics.Accuracy()
test_precision_metric = tf.keras.metrics.Precision()
test_recall_metric = tf.keras.metrics.Recall()
metrics = [train_loss_metric,train_accuracy_metric,train_precision_metric,train_recall_metric
            ,test_loss_metric,test_accuracy_metric,test_precision_metric,test_recall_metric]


print('==============================================')
print(f'Training starts for {NUM_EPOCH} epochs....')
print('==============================================')


for epoch in range(NUM_EPOCH):
    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCH} ===")
    reset_metrics(metrics)

    all_train_preds, all_train_labels = [], []

    print("-> Training phase")
    for batch_idx, (_, n, t, adj) in enumerate(dl.train_batch_iterator()):
        with tf.GradientTape() as tape:
            logits, loss, weigths = run_model(adj, n, t, training=True)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        y_true = tf.cast(tf.argmax(t, axis=-1) == 0, tf.float32)
        y_pred = tf.cast(tf.argmax(logits, axis=-1) == 0, tf.float32)
        y_mask = tf.cast(tf.argmax(t, axis=-1) == 0, tf.float32)

        train_loss_metric(loss)
        train_accuracy_metric(tf.argmax(t, axis=-1), tf.argmax(logits, axis=-1), sample_weight=weigths)
        train_precision_metric(y_mask, tf.cast(y_pred == 0, tf.float32))
        train_recall_metric(y_mask, tf.cast(y_pred == 0, tf.float32))

        all_train_preds.extend(y_pred.numpy())
        all_train_labels.extend(y_true.numpy())

        # 👇 Print per-batch metrics (optional)
        print(f"  Batch {batch_idx+1}: Loss={loss.numpy():.4f}, Accuracy={train_accuracy_metric.result().numpy():.4f}")

    print("-> Testing phase")
    all_test_preds, all_test_labels = [], []

    for batch_idx, (_, n, t, adj) in enumerate(dl.test_batch_iterator()):
        logits, loss, weigths = run_model(adj, n, t)

        y_true = tf.argmax(t, axis=-1).numpy()
        y_pred = tf.argmax(logits, axis=-1).numpy()
        y_mask = tf.cast(tf.argmax(t, axis=-1) == 0, tf.float32)

        test_loss_metric(loss)
        test_accuracy_metric(tf.argmax(t, axis=-1), tf.argmax(logits, axis=-1), sample_weight=weigths)
        test_precision_metric(y_mask, tf.cast(y_pred == 0, tf.float32))
        test_recall_metric(y_mask, tf.cast(y_pred == 0, tf.float32))

        all_test_preds.extend(y_pred)
        all_test_labels.extend(y_true)

        # 👇 Print per-batch test loss (optional)
        print(f"  Test Batch {batch_idx+1}: Loss={loss.numpy():.4f}, Accuracy={test_accuracy_metric.result().numpy():.4f}")

    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
    conf_matrix = confusion_matrix(all_test_labels, all_test_preds)

    # ✅ Print epoch summary
    print("== Epoch Summary ==")
    print("TRAIN Loss: {:.5f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(
        train_loss_metric.result().numpy(),
        train_accuracy_metric.result().numpy(),
        train_precision_metric.result().numpy(),
        train_recall_metric.result().numpy(),
        train_f1
    ))
    print("TEST  Loss: {:.5f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(
        test_loss_metric.result().numpy(),
        test_accuracy_metric.result().numpy(),
        test_precision_metric.result().numpy(),
        test_recall_metric.result().numpy(),
        test_f1
    ))
    print("Confusion Matrix:\n", conf_matrix)
    print("-" * 60)


# Save model weights
model.save_weights("outputs/evolve_gcn_model_weights.h5")
print("✅ Model weights saved to evolve_gcn_model_weights.h5")
