#  Téléchargement des Données Audio

## ⚠️ Note Importante

Le dataset audio complet n'est **pas inclus** dans le dépôt GitHub car les fichiers sont trop volumineux.

##  Comment télécharger les données ?

### Étape 1 : Télécharger depuis Google Drive

Cliquez sur le lien suivant pour télécharger le fichier ZIP contenant les données audio :

**🔗 [Télécharger le dataset (Google Drive)](https://drive.google.com/file/d/1mGvYGhzAnQzpgaxVaYhDRPSBZSP2rKIh/view?usp=drive_link)**

### Étape 2 : Extraire les fichiers

1. Une fois le téléchargement terminé, localisez le fichier `data.zip` sur votre ordinateur
2. Extrayez le contenu du fichier ZIP dans ce dossier `data/`

### Étape 3 : Structure attendue

Après extraction, la structure du dossier `data/` devrait ressembler à ceci :

```
data/
├── README.md (ce fichier)
├── CL_TR/          # Clean Training audio files
├── CL_TS/          # Clean Test audio files
├── N_TR/           # Noisy Training audio files
├── N_TS/           # Noisy Test audio files
└── processed/      # Processed data (généré automatiquement)
```

##  Vérification

Pour vérifier que tout est bien installé, assurez-vous que vous avez :
- ✓ Les dossiers `CL_TR`, `CL_TS`, `N_TR`, `N_TS` contenant les fichiers audio `.wav`
- ✓ Un nombre égal de fichiers clean et noisy correspondants

##  Problèmes ?

Si vous rencontrez des problèmes lors du téléchargement ou de l'extraction :
1. Vérifiez que vous avez assez d'espace disque disponible (~130 MB minimum)
2. Assurez-vous d'avoir accès au lien Google Drive
3. Réessayez le téléchargement si le fichier semble corrompu
