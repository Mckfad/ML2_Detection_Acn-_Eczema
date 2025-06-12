import gdown

# Télécharger automatiquement le modèle s’il n’existe pas déjà
@st.cache_resource
def download_model():
    file_id = "1omDDYKCmISTVfFMwUegsfMAH1i30o9aL"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "hybrid_skin_disease_model1.pth"
    
    if not os.path.exists(output):
        with st.spinner("Téléchargement du modèle depuis Google Drive..."):
            gdown.download(url, output, quiet=False)
    return output


import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torchvision.models as models
import os
import random

# Configuration de la page
st.set_page_config(
    page_title="Diagnostic Cutané Intelligent",
    page_icon="🩺",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.header {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}
.result-box {
    background-color: #e8f4fc;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    border-left: 5px solid #3498db;
}
.confidence-bar {
    height: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
    margin: 10px 0;
}
.confidence-fill {
    height: 100%;
    border-radius: 10px;
    background-color: #3498db;
    text-align: center;
    color: white;
    font-weight: bold;
}
.transform-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-top: 20px;
}
.transform-card {
    background: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Définition du modèle hybride
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Identity()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(1024 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes))
    
    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.efficientnet(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

# Fonction pour charger le modèle
@st.cache_resource
def load_model(model_path, num_classes):
    model = HybridModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Transformation des images (identique à mod__hybr.py avec visualisation)
class SkinDiseaseTransform:
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        self.augmentation = transforms.RandomHorizontalFlip(p=1.0)  # p=1.0 pour la démo
    
    def __call__(self, img, apply_augmentation=False):
        # Applique les transformations de base
        img_resized = img.resize((224, 224))
        
        # Applique l'augmentation si demandée (pour la démo)
        if apply_augmentation:
            img_flipped = self.augmentation(img_resized)
        else:
            img_flipped = img_resized
        
        # Conversion en tenseur et normalisation
        tensor_img = self.base_transform(img_flipped)
        
        return {
            'original': img,
            'resized': img_resized,
            'flipped': img_flipped,
            'tensor': tensor_img
        }

# Fonction pour visualiser les transformations
def show_transformations(transform_results):
    st.subheader("🔍 Prétraitements appliqués")
    
    with st.container():
        cols = st.columns(3)
        
        with cols[0]:
            st.image(transform_results['original'], caption="Image originale",use_container_width=True)
        
        with cols[1]:
            st.image(transform_results['resized'], caption=f"Redimensionnée (224x224)", use_container_width=True)
        
        with cols[2]:
            st.image(transform_results['flipped'], caption="Retournement horizontal",use_container_width=True)
    
    # Visualisation de la normalisation
    st.subheader("Normalisation des canaux RGB")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    tensor = transform_results['tensor']
    
    for i, (ax, channel, title) in enumerate(zip(
        axes,
        ['Rouge', 'Vert', 'Bleu'],
        ['Canal Rouge', 'Canal Vert', 'Canal Bleu']
    )):
        channel_data = tensor[i].numpy()
        im = ax.imshow(channel_data, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    st.pyplot(fig)

# Fonction de prédiction
def predict(image, model, class_names):
    transform = SkinDiseaseTransform()
    transformed = transform(image, apply_augmentation=random.random() > 0.5)  # 50% de chance d'appliquer le flip
    
    # Affiche les transformations
    show_transformations(transformed)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(transformed['tensor'].unsqueeze(0))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences = {class_names[i]: float(probabilities[0][i]) for i in range(len(class_names))}
    
    return confidences

# Interface principale
def main():
    st.title("🩺 Diagnostic Cutané Intelligent")
    st.markdown("""
    Cette application utilise une intelligence artificielle avancée pour aider à identifier les maladies de la peau 
    comme l'acné, l'eczéma et autres affections cutanées.
    """)
    
    # Chargement du modèle
    model_path = download_model()
    class_names = ["acne", "eczema"]  # Doit correspondre au modèle entraîné  
  
    # Section de téléchargement d'image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Charger une image")
        uploaded_file = st.file_uploader(
            "Téléversez une photo de la lésion cutanée",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléversée",use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("2. Résultats d'analyse")
            
            if st.button("Analyser l'image", type="primary"):
                with st.spinner("Analyse en cours..."):
                    try:
                        # Prédiction avec visualisation des transformations
                        confidences = predict(image, model, class_names)
                        
                        # Récupérer la classe prédite et sa confiance
                        predicted_class, max_confidence = max(confidences.items(), key=lambda x: x[1])
                        
                        # Si confiance < 95%, on affiche "Inconnue"
                        if max_confidence < 0.95:
                            display_class = "Inconnue"
                        else:
                            display_class = predicted_class.capitalize()
                        
                        # Affichage des résultats
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.subheader("Résultats de diagnostic")
                        
                        st.markdown(
                            f"**Diagnostic probable:** <span style='color:#3498db; font-size:1.2em'>{display_class}</span>", 
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**Confiance:** {max_confidence*100:.2f}%")
                        
                        st.subheader("Détail des probabilités:")
                        for class_name, confidence in confidences.items():
                            st.write(f"{class_name.capitalize()}")
                            st.markdown(f"""
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence*100}%">{confidence*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Conseils généraux seulement si confiance >= 95%
                        if max_confidence >= 0.95:
                            st.subheader("Conseils")
                            if predicted_class == "acne":
                                st.info("""
                                - Nettoyez doucement la peau 2 fois par jour
                                - Évitez de toucher ou percer les boutons
                                - Consultez un dermatologue si l'acné persiste
                                """)
                            elif predicted_class == "eczema":
                                st.info("""
                                - Hydratez régulièrement votre peau
                                - Évitez les savons agressifs
                                - Consultez un médecin pour des traitements topiques
                                """)
                        
                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de l'analyse: {str(e)}")
        else:
            st.info("Veuillez téléverser une image pour obtenir un diagnostic")

    # Section d'information
    with st.expander("ℹ️ À propos des prétraitements"):
        st.markdown("""
        **Transformations appliquées:**
        1. **Redimensionnement:** 224×224 pixels (taille attendue par le modèle)
        2. **Retournement horizontal:** Augmentation aléatoire des données
        3. **Normalisation:**
           - Moyenne: [0.485, 0.456, 0.406] (RGB)
           - Écart-type: [0.229, 0.224, 0.225] (RGB)
        
        **Pourquoi ces transformations?**
        - Identiques à celles utilisées pendant l'entraînement
        - Compatibles avec les modèles pré-entraînés (ImageNet)
        - Améliorent la généralisation du modèle
        """)

if __name__ == "__main__":
    main()
