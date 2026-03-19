📧 Spam Detection Engine (PyTorch & NLP)
Projet de Recherche Intensif – Summer School BIP (July 2025)
European University Cyprus (Larnaca, Chypre)

Ce projet a été développé dans le cadre d'un Blended Intensive Program (BIP) focalisé sur l'IA et la cybersécurité. L'objectif était de concevoir, en une semaine intensive, un classificateur de pointe capable de traiter des données textuelles tout en résolvant des problématiques de déséquilibre de classes.

📊 Visualisations & Résultats
Cette section présente les performances du modèle obtenues après l'entraînement.

1. Courbes d'Apprentissage (Loss & Accuracy)
Le graphique ci-dessous montre l'évolution de la fonction de perte (Focal Loss) et de la précision au fil des époques. On observe une convergence rapide et une excellente gestion de l'overfitting grâce au Dropout et au Weight Decay.

<img width="1616" height="600" alt="image" src="https://github.com/user-attachments/assets/f500c5fc-36e6-440a-b523-a777c0ccadf3" />



2. Matrice de Confusion
La matrice de confusion confirme la robustesse du modèle. Avec une Accuracy de 99,03%, le moteur distingue quasi parfaitement les messages légitimes (Ham) des messages malveillants (Spam), minimisant les faux positifs.

<img width="735" height="386" alt="image" src="https://github.com/user-attachments/assets/747e6607-28e5-46d5-839f-b7cc91ab6e02" />


<img width="616" height="519" alt="image" src="https://github.com/user-attachments/assets/b7a402da-dcba-4d51-973c-4c10aaa1e263" />


🚀 Points Forts Techniques
Format Intensif (BIP) : Développement et optimisation réalisés en juillet 2025 lors d'une école d'été internationale.

Architecture PyTorch Custom : Implémentation utilisant les classes nn.Module, Dataset et DataLoader.

Focal Loss Implementation : Fonction de perte sur mesure pour gérer le déséquilibre des données (Spam vs Ham).

Régularisation Avancée : Utilisation de couches Dropout et d'une pénalité L2 pour garantir la portabilité du modèle.

Early Stopping : Monitoring dynamique pour arrêter l'entraînement au point optimal de performance.

🛠️ Stack Technologique
Langage : Python

Deep Learning : PyTorch

Traitement de données : Pandas, Scikit-Learn

NLP : CountVectorizer (Bag-of-Words), Regex Tokenization

Visualisation : Seaborn, Matplotlib

⚙️ Structure du Pipeline
Preprocessing : Tokenisation et vectorisation des courriels.

Data Loading : Pipeline de tenseurs optimisé pour l'entraînement.

Modeling : Modèle de régression logistique.

Evaluation : Analyse fine via F1-Score et Matrice de confusion.

Contexte International
Ce travail a été effectué lors du programme BIP à l'European University Cyprus, regroupant des étudiants et chercheurs européens autour des enjeux de l'IA appliquée à la cybersécurité.

Contact :
Anas ISARTI - LinkedIn
