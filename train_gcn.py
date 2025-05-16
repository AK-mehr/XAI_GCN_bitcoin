import numpy as np
import tensorflow as tf
import argparse


from data_loader import EllipticDatasetLoader
from models import GCNSequential
from models.layers import GCNLayer


print('hello')

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
parser.add_argument("--num_epoch", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=1e-3)

args = parser.parse_args()

# Assign values from args
DATADIR = args.datadir_path
FILTER_UNKNOWN = args.filter_unknown
ONLY_LOCAL_FEATURE = args.local_features_only
CLASS_WEIGTHS = args.class_weights
TEST_SHARE = args.test_portion
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


model = GCNSequential([
    GCNLayer(64,"relu"),
    tf.keras.layers.Dropout(0.3),
    GCNLayer(dl.num_classes)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# FOCAL LOSS
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def run_model(adj,nodes,targets,training=False):
    weigths = tf.reduce_sum(CLASS_WEIGTHS * targets, axis=-1)
    logits = model([adj, nodes],training=training)
    loss = loss_func(targets, logits,sample_weight=weigths)
    return logits, loss, weigths


train_loss_metric = tf.keras.metrics.Mean()
train_accuracy_metric = tf.keras.metrics.Accuracy()
train_precision_metric = tf.keras.metrics.Precision()
train_recall_metric = tf.keras.metrics.Recall()
test_loss_metric = tf.keras.metrics.Mean()
test_accuracy_metric = tf.keras.metrics.Accuracy()
test_precision_metric = tf.keras.metrics.Precision()
test_recall_metric = tf.keras.metrics.Recall()
#CONFUSION MATRIX
metrics = [train_loss_metric,train_accuracy_metric,train_precision_metric,train_recall_metric
            ,test_loss_metric,test_accuracy_metric,test_precision_metric,test_recall_metric]

print('==============================================')
print(f'Training starts for {NUM_EPOCH} epochs....')
print('==============================================')

for epoch in range(NUM_EPOCH):
    reset_metrics(metrics)

    for _, n, t, adj in dl.train_batch_iterator():
        with tf.GradientTape() as tape:
            logits, loss, weigths = run_model(adj, n, t, training=True)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights))

        y_true = tf.cast(tf.argmax(t,axis=-1) == 0,tf.float32)
        y_pred = tf.cast(tf.argmax(logits,axis=-1) == 0,tf.float32)
        train_loss_metric(loss)
        train_accuracy_metric(tf.argmax(t,axis=-1), tf.argmax(logits,axis=-1), sample_weight=weigths)
        train_precision_metric(y_true, y_pred)
        train_recall_metric(y_true, y_pred)

    for _, n, t, adj in dl.test_batch_iterator():
        logits, loss, weigths = run_model(adj, n, t)

        y_true = tf.cast(tf.argmax(t, axis=-1) == 0, tf.float32)
        y_pred = tf.cast(tf.argmax(logits, axis=-1) == 0, tf.float32)
        test_loss_metric(loss)
        test_accuracy_metric(tf.argmax(t,axis=-1), tf.argmax(logits,axis=-1), sample_weight=weigths)
        test_precision_metric(y_true, y_pred)
        test_recall_metric(y_true, y_pred)

    print("Epoch: {}\nTRAIN Loss: {:.5}| Accuracy: {:.4}| Precision: {:.4}| Recall: {:.4}\nTEST Loss: {:.4}| Accuracy: {:.4}| Precision: {:.4}| Recall: {:.4}".format(
        epoch, train_loss_metric.result().numpy(), train_accuracy_metric.result().numpy(),
        train_precision_metric.result().numpy(),train_recall_metric.result().numpy(),
        test_loss_metric.result().numpy(), test_accuracy_metric.result().numpy(),
        test_precision_metric.result().numpy(),test_recall_metric.result().numpy()
    ))

# Save model weights
model.save_weights("outputs/gcn_model_weights.h5")
print("✅ Model weights saved to gcn_model_weights.h5")

