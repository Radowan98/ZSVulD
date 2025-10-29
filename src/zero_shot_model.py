"""
Zero-Shot Cross-Project Vulnerability Detection
------------------------------------------------
Implements the full model used in:
"A Zero-Shot Framework for Cross-Project Vulnerability Detection in Source Code"

Supports three experimental configurations:
1. Source: Devign (Qemu + FFmpeg) → Target: ReVeal (Debian + Chrome)
2. Source: Qemu + ReVeal (Debian + Chrome) → Target: FFmpeg
3. Source: Devign (Qemu + FFmpeg) + Chrome → Target: Debian
"""

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    roc_auc_score, f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


# =========================================================
# Helper functions
# =========================================================

def create_neural_network(input_dim: int):
    """Defines the base neural network model."""
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def semi_supervised_transfer_learning(x_train, y_train, x_test, class_weight_dict):
    """Train on source data, pseudo-label target, and retrain."""
    model = create_neural_network(x_train.shape[1])
    model.fit(
        x_train, y_train,
        epochs=50, batch_size=64,
        validation_split=0.2,
        class_weight=class_weight_dict,
        verbose=0
    )

    # Predict pseudo-labels on target
    y_pred = model.predict(x_test, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()

    # Augment source with pseudo-labelled target
    x_train_aug = np.concatenate((x_train, x_test))
    y_train_aug = np.concatenate((y_train, y_pred_binary))

    model.fit(
        x_train_aug, y_train_aug,
        epochs=30, batch_size=32,
        class_weight=class_weight_dict,
        verbose=0
    )

    return model, y_pred.flatten()


def train_base_model_xgb(x_train, y_train, sample_weights=None):
    """Train XGBoost ensemble with optional weighting."""
    model = XGBClassifier(
        n_estimators=100, max_depth=3,
        use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(x_train, y_train, sample_weight=sample_weights)
    return model


# =========================================================
# Experiment configuration
# =========================================================

def load_embeddings(setting):
    """Load embeddings and labels based on selected experimental setting."""

    # Load all
    qemu = np.load('dataset/qemu_embeddings.npy')
    ffm = np.load('dataset/ffm_embeddings.npy')
    deb = np.load('dataset/deb_embeddings.npy')
    chrr = np.load('dataset/chr_embeddings.npy')

    qemu_l = np.load('dataset/qemu_labels.npy')
    ffm_l = np.load('dataset/ffm_labels.npy')
    deb_l = np.load('dataset/deb_labels.npy')
    chrr_l = np.load('dataset/chr_labels.npy')

    if setting == 1:
        # Devign → ReVeal
        source_embeddings = np.concatenate([qemu, ffm])
        source_labels = np.concatenate([qemu_l, ffm_l])
        target_embeddings = np.concatenate([deb, chrr])
        target_labels = np.concatenate([deb_l, chrr_l])
        desc = "Setting 1: Devign (Qemu+FFmpeg) → ReVeal (Debian+Chrome)"

    elif setting == 2:
        # Qemu + ReVeal → FFmpeg
        source_embeddings = np.concatenate([qemu, deb, chrr])
        source_labels = np.concatenate([qemu_l, deb_l, chrr_l])
        target_embeddings = ffm
        target_labels = ffm_l
        desc = "Setting 2: Qemu+ReVeal (Debian+Chrome) → FFmpeg"

    elif setting == 3:
        # Devign + Chrome → Debian
        source_embeddings = np.concatenate([qemu, ffm, chrr])
        source_labels = np.concatenate([qemu_l, ffm_l, chrr_l])
        target_embeddings = deb
        target_labels = deb_l
        desc = "Setting 3: Devign (Qemu+FFmpeg)+Chrome → Debian"

    else:
        raise ValueError("Invalid setting. Choose 1, 2, or 3.")

    print(f"\n=== {desc} ===")
    return source_embeddings, source_labels, target_embeddings, target_labels, desc


# =========================================================
# Main
# =========================================================

def main(setting):
    # Load datasets
    src_emb, src_lbl, tgt_emb, tgt_lbl, desc = load_embeddings(setting)

    # Normalise
    scaler = StandardScaler()
    src_emb = scaler.fit_transform(src_emb)
    tgt_emb = scaler.transform(tgt_emb)

    # Compute class weights
    classes = np.unique(src_lbl)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=src_lbl)
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}

    # Ensemble setup
    num_iterations = 5
    ensemble_predictions = np.zeros(len(tgt_lbl))

    for i in range(num_iterations):
        print(f"\n[Iteration {i+1}/{num_iterations}] Training semi-supervised model...")
        model_nn, y_pred_nn = semi_supervised_transfer_learning(src_emb, src_lbl, tgt_emb, class_weight_dict)

        y_train_pred = model_nn.predict(src_emb, verbose=0).flatten()
        sample_weights = np.where(src_lbl == 1, y_train_pred, 1 - y_train_pred)

        model_xgb = train_base_model_xgb(src_emb, src_lbl, sample_weights)
        model_xgb_pred = model_xgb.predict(tgt_emb)
        ensemble_predictions += model_xgb_pred

    # Average ensemble
    ensemble_avg = ensemble_predictions / num_iterations
    y_pred_final = (ensemble_avg > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(tgt_lbl, y_pred_final)
    recall = recall_score(tgt_lbl, y_pred_final)
    precision = precision_score(tgt_lbl, y_pred_final)
    auc = roc_auc_score(tgt_lbl, ensemble_avg)
    f1 = f1_score(tgt_lbl, y_pred_final)
    tn, fp, fn, tp = confusion_matrix(tgt_lbl, y_pred_final).ravel()
    g_mean = np.sqrt(tp / (tp + fn))
    pf = fp / (fp + tn)

    print("\n=== Evaluation Results ===")
    print(f"{desc}")
    print(f"Accuracy:   {accuracy:.3f}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f"F1-score:   {f1:.3f}")
    print(f"AUC:        {auc:.3f}")
    print(f"G-mean:     {g_mean:.3f}")
    print(f"PF value:   {pf:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Vulnerability Detection Framework")
    parser.add_argument("--setting", type=int, required=True, choices=[1, 2, 3],
                        help="Experimental setting (1, 2, or 3)")
    args = parser.parse_args()
    main(args.setting)
