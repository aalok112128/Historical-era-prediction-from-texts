import numpy as np
import pandas as pd
import pickle
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load saved features and labels ──────────────────────────
print("Loading features and labels...")
X      = scipy.sparse.load_npz("features.npz")
labels = np.load("labels.npy", allow_pickle=True)
print(f"Features shape : {X.shape}")
print(f"Total samples  : {len(labels)}")

# ── Train / Test split ───────────────────────────────────────
# 80% training, 20% testing
# stratify = keeps era proportions balanced in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

results = {}   # store accuracy + F1 for each model


# ════════════════════════════════════════════════════════════
# MODEL 1 — NAIVE BAYES (ComplementNB)
# ════════════════════════════════════════════════════════════
# ComplementNB handles class imbalance better than MultinomialNB
# Works well with our stylometric features too

print("\n" + "="*50)
print("MODEL 1 — Naive Bayes")
print("="*50)

nb_model = ComplementNB(alpha=0.1)
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

nb_acc = accuracy_score(y_test, nb_preds)
nb_f1  = f1_score(y_test, nb_preds, average='weighted')
results['Naive Bayes'] = {'Accuracy': nb_acc, 'F1 Score': nb_f1}

print(f"Accuracy : {nb_acc*100:.2f}%")
print(f"F1 Score : {nb_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, nb_preds))


# ════════════════════════════════════════════════════════════
# MODEL 2 — SVM (Support Vector Machine)
# ════════════════════════════════════════════════════════════
# Usually the strongest model for text classification
# class_weight='balanced' fixes our class imbalance issue

print("\n" + "="*50)
print("MODEL 2 — SVM")
print("="*50)

svm_model = LinearSVC(
    C=1.0,
    max_iter=10000,
    dual=False,
    class_weight='balanced'   # handles Renaissance having fewer samples
)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_preds)
svm_f1  = f1_score(y_test, svm_preds, average='weighted')
results['SVM'] = {'Accuracy': svm_acc, 'F1 Score': svm_f1}

print(f"Accuracy : {svm_acc*100:.2f}%")
print(f"F1 Score : {svm_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, svm_preds))


# ════════════════════════════════════════════════════════════
# MODEL 3 — RANDOM FOREST
# ════════════════════════════════════════════════════════════
# Builds 200 decision trees and takes majority vote
# Handles stylometric features especially well

print("\n" + "="*50)
print("MODEL 3 — Random Forest")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1              # use all CPU cores
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
rf_f1  = f1_score(y_test, rf_preds, average='weighted')
results['Random Forest'] = {'Accuracy': rf_acc, 'F1 Score': rf_f1}

print(f"Accuracy : {rf_acc*100:.2f}%")
print(f"F1 Score : {rf_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, rf_preds))


# ════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
results_df = pd.DataFrame(results).T.sort_values('Accuracy', ascending=False)
print(results_df.round(4))

best_name = results_df.index[0]
best_acc  = results_df.loc[best_name, 'Accuracy']
print(f"\nBest model : {best_name} ({best_acc*100:.2f}%)")


# ════════════════════════════════════════════════════════════
# CONFUSION MATRICES — saved as images
# ════════════════════════════════════════════════════════════
eras        = ['Enlightenment', 'Renaissance', 'Romantic', 'Victorian']
model_preds = {
    'Naive Bayes'   : nb_preds,
    'SVM'           : svm_preds,
    'Random Forest' : rf_preds,
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices — All 3 Models', fontsize=14, fontweight='bold')

for ax, (name, preds) in zip(axes, model_preds.items()):
    cm = confusion_matrix(y_test, preds, labels=eras)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=eras, yticklabels=eras, ax=ax)
    ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_test, preds)*100:.1f}%')
    ax.set_xlabel('Predicted Era')
    ax.set_ylabel('Actual Era')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print("\nSaved: confusion_matrices.png")

# ════════════════════════════════════════════════════════════
# ACCURACY BAR CHART — saved as image
# ════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(8, 5))
colors = ['#534AB7', '#1D9E75', '#D85A30']
bars   = ax2.bar(results_df.index, results_df['Accuracy'] * 100, color=colors)

for bar, val in zip(bars, results_df['Accuracy'] * 100):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.1f}%',
             ha='center', va='bottom', fontweight='bold')

ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Model Accuracy Comparison', fontweight='bold')
ax2.set_ylim(0, 105)
ax2.axhline(y=25, color='red', linestyle='--',
            alpha=0.5, label='Random baseline (25%)')
ax2.legend()
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: accuracy_comparison.png")


# ════════════════════════════════════════════════════════════
# SAVE BEST MODEL
# ════════════════════════════════════════════════════════════
best_model_obj = {
    'Naive Bayes'   : nb_model,
    'SVM'           : svm_model,
    'Random Forest' : rf_model,
}[best_name]

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model_obj, f)

with open("best_model_name.txt", "w") as f:
    f.write(best_name)

print(f"\nSaved: best_model.pkl ({best_name})")
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print("Files created:")
print("  best_model.pkl         — best trained model")
print("  best_model_name.txt    — name of best model")
print("  confusion_matrices.png — visual evaluation")
print("  accuracy_comparison.png — bar chart")
print("\nNext step: BERT model (innovation layer)")

import json

# Save results to file so report can load them automatically
results_to_save = {
    "Naive Bayes"   : {
        "Accuracy": round(nb_acc * 100, 2),
        "F1 Score": round(nb_f1, 4)
    },
    "SVM"           : {
        "Accuracy": round(svm_acc * 100, 2),
        "F1 Score": round(svm_f1, 4)
    },
    "Random Forest" : {
        "Accuracy": round(rf_acc * 100, 2),
        "F1 Score": round(rf_f1, 4)
    },
}

with open("results.json", "w") as f:
    json.dump(results_to_save, f, indent=2)

print("Saved: results.json")