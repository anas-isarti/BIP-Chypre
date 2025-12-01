# BIP-Chypre
import pandas as pd
import numpy as np # Ajout de numpy pour les opérations sur les tableaux
import matplotlib.pyplot as plt
import seaborn as sns # Importation de seaborn pour les heatmaps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# --- 1. LECTURE ET EXPLORATION DES DONNÉES ---
url = 'spam_ham_dataset.csv'
df = pd.read_csv(url)

print("--- df.info() ---")
df.info()
print("\n--- df.shape ---")
print(df.shape)
print("\n--- df.head(5) ---")
print(df.head(5))
print("\n--- df.isnull().sum() ---")
print(df.isnull().sum())
print("\n--- df.describe() ---")
print(df.describe())

# Calculate the ham and spam occurances
value_counts = df['label'].value_counts()
percentages = (value_counts / len(df)) * 100
print("\n--- Ham/Spam Value Counts with Percentages ---")
print(percentages.round(2))

# Plot the fire occurances as a pie chart
print("\n--- Affichage du graphique circulaire ---")
plt.figure(figsize=(5, 5))
plt.pie(percentages, labels=percentages.index, autopct='%1.2f%%', startangle=90)
plt.title('Distribution des labels Ham vs Spam')
plt.show()


# --- 2. PRÉPARATION DES DONNÉES ET VECTORISATION  ---
# Préparation des données
X = df["text"]
y = df["label_num"]

# Découpage en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# --- 3. CRÉATION D'UN JEU DE VALIDATION  ---
# On divise les données d'entraînement pour créer un jeu de validation.
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
)


# --- 4. ENTRAÎNEMENT DES MODÈLES  ---

# 4.1. Entraînement du modèle LogisticRegression
print("\n=== Entraînement de la Régression Logistique ===")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_final, y_train_final) # Entraîné sur le jeu final pour être comparable
y_pred_logreg = logreg.predict(X_test_tfidf)

# 4.2. Entraînement du modèle XGBoost avec suivi
print("\n=== Entraînement de XGBoost avec suivi de la validation ===")
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=500,
    random_state=42
)
eval_set = [(X_train_final, y_train_final), (X_val, y_val)]
xgb_model.fit(X_train_final, y_train_final, eval_set=eval_set, verbose=False)
y_pred_xgb = xgb_model.predict(X_test_tfidf)

# Récupération des résultats pour le graphique XGBoost
results_xgb = xgb_model.evals_result()
train_loss_xgb = results_xgb['validation_0']['logloss']
val_loss_xgb = results_xgb['validation_1']['logloss']
epochs_xgb = range(1, len(train_loss_xgb) + 1)

# 4.3. Entraînement du modèle MLP avec suivi
print("\n=== Entraînement du MLP avec suivi de la validation ===")
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10
)
mlp.fit(X_train_tfidf, y_train)
y_pred_mlp = mlp.predict(X_test_tfidf)

# Récupération des résultats pour le graphique MLP
train_loss_mlp = mlp.loss_curve_
val_error_mlp = 1 - np.array(mlp.validation_scores_)
epochs_mlp = range(1, len(train_loss_mlp) + 1)


# --- 5. ÉVALUATION DES PERFORMANCES  ---
print("\n\n--- RÉSULTATS DE L'ÉVALUATION FINALE SUR LE JEU DE TEST ---")
print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))

print("\n=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

print("\n=== MLP Neural Network ===")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))


# --- 6. VISUALISATION DES COURBES D'APPRENTISSAGE  ---
print("\n--- Affichage des courbes d'apprentissage ---")
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Courbes d\'Apprentissage (Training Loss vs Validation Loss)', fontsize=16)

# Graphique pour XGBoost
axes[0].plot(epochs_xgb, train_loss_xgb, 'b-', label='Training Loss')
axes[0].plot(epochs_xgb, val_loss_xgb, 'r-', label='Validation Loss')
axes[0].set_title('XGBoost')
axes[0].set_xlabel('Nombre d\'arbres (Boosting Rounds)')
axes[0].set_ylabel('Log Loss')
axes[0].legend()
axes[0].grid(True)

# Graphique pour MLP
axes[1].plot(epochs_mlp, train_loss_mlp, 'b-', label='Training Loss')
axes[1].plot(epochs_mlp, val_error_mlp, 'r-', label='Validation Error')
axes[1].set_title('Réseau de Neurones (MLP)')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss / Error')
axes[1].legend()
axes[1].grid(True)
plt.show()


# --- 7. VISUALISATION AVEC HEATMAPS  ---
print("\n--- Affichage des matrices de confusion ---")
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.suptitle('Matrices de Confusion pour chaque Modèle (sur le jeu de test)', fontsize=16)

labels = ['Ham (Vrai)', 'Spam (Vrai)']
predicted_labels = ['Ham (Prédit)', 'Spam (Prédit)']

sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=predicted_labels, yticklabels=labels)
axes[0].set_title('Régression Logistique')
axes[0].set_ylabel('Label Réel')
axes[0].set_xlabel('Label Prédit')

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=predicted_labels, yticklabels=labels)
axes[1].set_title('XGBoost')
axes[1].set_xlabel('Label Prédit')

sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
            xticklabels=predicted_labels, yticklabels=labels)
axes[2].set_title('Réseau de Neurones (MLP)')
axes[2].set_xlabel('Label Prédit')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
