from pathlib import Path
from datetime import datetime
import json, sqlite3, uuid
import numpy as np
import streamlit as st
from PIL import Image

BASE = Path(__file__).parent
DATA = BASE / "data"
SCANS = DATA / "scans"
MODEL = BASE / "model" / "plantguard_best.keras"
LABELS = BASE / "model" / "class_names.json"
REPORT = BASE / "model" / "training_report.json"
for p in [SCANS, DATA / "history", MODEL.parent]: p.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="PlantGuard AI", page_icon="🌿", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
.stApp{background:radial-gradient(circle at 80% 0%,#173326 0,#0b1210 42%);color:#edf5ef}
[data-testid="stSidebar"]{background:#0d1713;border-right:1px solid #263a30}
h1,h2,h3{font-family:Manrope,sans-serif;letter-spacing:-.03em}
.hero{background:linear-gradient(120deg,#18372a,#12231b);border:1px solid #315642;padding:28px;border-radius:22px;margin-bottom:18px}
.kicker{color:#91e1b2;font-size:12px;font-weight:700;letter-spacing:.16em}
.hero h1{color:#f2fff5;font-size:38px;margin:8px 0}
.hero p,.muted{color:#bed1c4}
.metric{background:#14221b;border:1px solid #2b4335;padding:17px;border-radius:16px}
.metric small{color:#a8bdb0;text-transform:uppercase;letter-spacing:.08em}
.metric strong{display:block;color:#f2fff5;font-size:27px;margin-top:5px}
div.stButton>button{background:#78d6a0;color:#092014;border:0;border-radius:10px;font-weight:700}
div.stButton>button:hover{background:#9be8b9;color:#092014}
[data-testid="stFileUploader"]{background:#13221a;border:1px dashed #466c54;border-radius:14px}
</style>
""", unsafe_allow_html=True)

def connect():
    c=sqlite3.connect(DATA/"plantguard_history.db")
    c.execute("""CREATE TABLE IF NOT EXISTS scans(
    id TEXT PRIMARY KEY, created TEXT, plant TEXT, status TEXT, disease TEXT,
    confidence REAL, image_path TEXT)""")
    c.commit()
    return c

def history():
    with connect() as c:
        return c.execute("SELECT id,created,plant,status,disease,confidence,image_path FROM scans ORDER BY created DESC").fetchall()

@st.cache_resource
def load_model():
    if not MODEL.exists() or not LABELS.exists(): return None, None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL), json.loads(LABELS.read_text())
    except Exception as e:
        return None, {"error":str(e)}

def classify(img):
    model, labels = load_model()

    if model is None:
        error = (
            "Model loading failed: " + labels["error"]
            if labels and "error" in labels
            else "Trained model files are missing."
        )
        return None, error

    # Convert image to RGB and resize
    img = img.convert("RGB").resize((224, 224))

    # Convert image to an array (0–255)
    x = np.asarray(img, dtype=np.float32)

    # Add batch dimension
    x = np.expand_dims(x, axis=0)

    # Predict
    probs = model.predict(x, verbose=0)[0]

    # Get highest probability
    i = int(np.argmax(probs))
    label = labels[i]

    # Separate plant and disease
    plant, disease = label.split("___", 1)

    plant = plant.replace("_", " ").replace(",", "")
    disease = disease.replace("_", " ").strip()

    # Determine health status
    status = (
        "Healthy"
        if "healthy" in disease.lower()
        else "Possible disease detected"
    )

    return {
        "plant": plant,
        "disease": disease,
        "status": status,
        "confidence": float(probs[i])
    }, None
    
def save(img,result):
    sid=str(uuid.uuid4()); path=SCANS/f"{sid}.jpg"
    img.convert("RGB").save(path,"JPEG",quality=88)
    with connect() as c:
        c.execute("INSERT INTO scans VALUES(?,?,?,?,?,?,?)",
        (sid,datetime.now().astimezone().isoformat(timespec="seconds"),result["plant"],
         result["status"],result["disease"],result["confidence"],str(path.relative_to(BASE))))

def metric(label,value):
    st.markdown(f'<div class="metric"><small>{label}</small><strong>{value}</strong></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌿 PlantGuard AI")
    st.caption("PLANT HEALTH INTELLIGENCE")
    page=st.radio("NAVIGATION",["Dashboard","Plant Scanner","Scan History","Model Insights"])
    st.divider()
    st.markdown("🟢 Website ready  \n🟢 Camera & upload  \n🟢 Local history")
    st.markdown(
    "🟢 Model connected"
    if MODEL.exists() and LABELS.exists()
    else "🟡 Model files missing"
)

if page=="Dashboard":
    st.markdown('<div class="hero"><div class="kicker">PLANT HEALTH, MADE SIMPLE</div><h1>A smarter way to care for your plants.</h1><p>Capture a leaf, identify its plant category and check its health with the PlantGuard AI image classifier.</p></div>',unsafe_allow_html=True)
    rows=history(); healthy=sum(r[3]=="Healthy" for r in rows); sick=sum(r[3]=="Possible disease detected" for r in rows)
    a,b,c=st.columns(3)
    with a: metric("TOTAL SCANS",len(rows))
    with b: metric("HEALTHY RESULTS",healthy)
    with c: metric("POSSIBLE DISEASE",sick)
    st.markdown("### Get started")
    st.write("Use Plant Scanner to take a photo or upload an image. Results become available after the trained model is installed.")
    if MODEL.exists(): st.success("Trained model files detected.")
    else: st.info("Portal is ready. The disease classifier will activate after model training.")
    st.markdown("### Recent activity")
    for r in rows[:5]: st.write(f"**{r[2]}** · {r[3]} · {r[1][:16]}")
    if not rows: st.caption("Your saved scans will appear here.")

elif page=="Plant Scanner":
    st.header("📷 Plant Scanner")
    st.caption("Photograph one leaf clearly in natural light.")
    a,b=st.columns(2)
    with a: camera=st.camera_input("Take a plant photo")
    with b: upload=st.file_uploader("Or upload an image",type=["jpg","jpeg","png","webp"])
    chosen=camera if camera is not None else upload
    if chosen:
        img=Image.open(chosen).convert("RGB")
        st.image(img,caption="Selected image",use_container_width=True)
        if st.button("Analyze plant",type="primary",use_container_width=True):
            with st.spinner("Analyzing image..."): result,error=classify(img)
            if error:
                st.warning(error)
                st.info("No disease label has been guessed. Add the trained model before using AI diagnosis.")
            else:
                x,y,z=st.columns(3)
                with x: metric("PLANT CATEGORY",result["plant"])
                with y: metric("HEALTH STATUS",result["status"])
                with z: metric("CONFIDENCE",f'{result["confidence"]*100:.1f}%')
                st.write("**Predicted condition:**",result["disease"])
                if result["confidence"]<.60: st.warning("Low confidence. Retake the photo or seek another opinion.")
                save(img,result); st.success("Saved to local scan history.")
    else:
        st.info("Tips: keep the leaf in focus, avoid harsh shadows, and include the affected area.")

elif page=="Scan History":
    st.header("🕘 Scan History")
    rows=history()
    if not rows: st.info("No scans saved yet. Visit Plant Scanner to begin.")
    else:
        q=st.text_input("Search plant or result")
        shown=[r for r in rows if q.lower() in " ".join(map(str,r[2:5])).lower()] if q else rows
        st.caption(f"{len(shown)} saved scan(s)")
        for r in shown:
            with st.container(border=True):
                x,y=st.columns([1,3]); path=BASE/r[6]
                with x:
                    if path.exists(): st.image(str(path),use_container_width=True)
                with y:
                    st.markdown(f"**{r[2]}**")
                    st.write(r[3],"·",r[4])
                    st.caption(f'{r[1]} · Confidence: {r[5]*100:.1f}%')
        if st.button("Delete all local history"):
            with connect() as c: c.execute("DELETE FROM scans")
            for p in SCANS.glob("*.jpg"): p.unlink(missing_ok=True)
            st.success("History cleared."); st.rerun()
    st.caption("Local SQLite and image files can be lost when the cloud app restarts or redeploys. This is prototype storage, not permanent backup.")

else:
    st.header("🧠 Model Insights")
    if REPORT.exists():
        report=json.loads(REPORT.read_text())
        x,y,z=st.columns(3)
        with x: metric("TEST ACCURACY",f'{report.get("test_accuracy",0)*100:.1f}%')
        with y: metric("CLASSES",report.get("num_classes","—"))
        with z: metric("MODEL","MobileNetV2")
        st.write("Training date:",report.get("trained_at","Not recorded"))
        st.write("Dataset classes:",", ".join(report.get("class_names",[])))
    else: st.info("No training report yet. Run the Colab training script to create model files and metrics.")
    st.markdown("### Hybrid workflow")
    st.write("1. MobileNetV2 classifies the combined plant/disease label.")
    st.write("2. A rule-based interpretation extracts the plant category and marks labels containing 'healthy' as healthy.")
    st.write("3. Test accuracy, classification report and confusion matrix are exported after training.")
    st.warning("The model only recognizes classes represented in its labelled training dataset. Predictions are estimates, not definitive diagnoses.")
