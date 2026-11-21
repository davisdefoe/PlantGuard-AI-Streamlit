# PlantGuard AI — Streamlit Edition

Dark-themed plant-health portal with camera capture, image upload, plant/disease classification, local SQLite history, and model metrics.

## Important
The website UI and local scan-history system work immediately. The AI does **not** make up a result when no trained model is present. The scanner clearly says the classifier is awaiting training until you add the exported model files.

## A. Upload the project using a browser
1. Extract this ZIP on your computer.
2. Visit https://github.com/new and create a repository named `PlantGuard-AI-Streamlit`.
3. Open the new repository and choose **Add file → Upload files**.
4. Upload the contents of the extracted folder (not the outer folder).
5. Commit to `main`. Confirm `app.py` and `requirements.txt` are visible at the repository root.

## B. Deploy free
1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Click **Create app** and select your repository.
3. Branch: `main`; main file path: `app.py`.
4. Deploy. Your public app gets a `streamlit.app` URL.

## C. Train the model alongside the website
Use Google Colab: https://colab.research.google.com/
1. Upload `train_model.py` to a new Colab notebook session.
2. Install: `!pip install tensorflow==2.15.1 scikit-learn==1.5.2`
3. Upload a labelled dataset in this exact structure:

dataset/
  train/
    Tomato___Early_blight/
    Tomato___healthy/
    Potato___Early_blight/
  validation/
    Tomato___Early_blight/
    Tomato___healthy/
    Potato___Early_blight/
  test/
    Tomato___Early_blight/
    Tomato___healthy/
    Potato___Early_blight/

Each class folder contains its own images. Include the same class folders in all three splits. Keep test images separate from training images.
4. Run `!python train_model.py`.
5. Download the generated `models` folder containing `plantguard_model.keras`, `labels.json`, and `training_report.json`.
6. Upload those three files to the `models/` folder in GitHub and commit.
7. To enable inference in the hosted app, add the line `tensorflow-cpu==2.15.1` to `requirements.txt`, then open the Streamlit app's **Advanced settings** and select **Python 3.11** before redeploying. TensorFlow is intentionally excluded from the first lightweight deployment so the portal can launch before the model is ready.

## Model design
MobileNetV2 transfer learning + fine-tuning classifies combined labels such as `Tomato___Early_blight`. The app parses the plant name and disease label. This is a practical first classifier, not universal plant recognition: it only knows classes included in the dataset. The exported report includes held-out test accuracy, classification report and confusion matrix.

## Storage warning
The selected local-history mode saves records in `data/plantguard_history.db` and images in `data/scans/`. Streamlit Community Cloud local files are not durable and may be lost during restart/redeploy. Do not treat this as permanent backup. For persistent history, a cloud database and object storage would be needed later.

## Free-tier note
Streamlit Community Cloud is free subject to its current limits. Colab's free compute is not guaranteed and can disconnect or limit sessions. No paid API is required by this project, but free services can change their quotas.

## Files
- `app.py` — Streamlit portal
- `train_model.py` — Colab training script
- `requirements.txt` — lightweight deployment dependencies
- `requirements-training.txt` — model-training dependencies
- `.streamlit/config.toml` — dark theme
- `models/` — exported model files go here
- `data/` — local scan history and images
