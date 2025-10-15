import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

# =============== LOAD MODEL ===============
model = tf.keras.models.load_model('image_classifier_model.h5')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# =============== PAGE CONFIG ===============
st.set_page_config(page_title="AI Image Classifier", page_icon="🤖", layout="centered")

# =============== DARK MODE STYLE ===============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

body {
    background-color: #0B0E11;
    color: #E0E6ED;
    font-family: 'Poppins', sans-serif;
}

/* Navbar */
.navbar {
    background-color: #12151B;
    padding: 12px 30px;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 0 20px rgba(0, 191, 255, 0.15);
}
.navbar a {
    color: #AAB2BF;
    text-decoration: none;
    margin-right: 25px;
    transition: 0.3s;
}
.navbar a:hover {
    color: #00BFFF;
    text-shadow: 0 0 8px #00BFFF;
}
.brand {
    color: #00BFFF;
    font-weight: 700;
    font-size: 22px;
}

/* Sections */
.section {
    background-color: #12151B;
    border-radius: 18px;
    padding: 30px;
    margin-top: 25px;
    box-shadow: 0 0 25px rgba(0, 191, 255, 0.05);
    transition: 0.3s;
}
.section:hover {
    box-shadow: 0 0 25px rgba(0, 191, 255, 0.2);
    transform: scale(1.01);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #0096FF, #006BCE);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px 28px;
    border: none;
    transition: 0.3s;
    box-shadow: 0 0 10px rgba(0,191,255,0.2);
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00C8FF, #0096FF);
    box-shadow: 0 0 20px rgba(0,191,255,0.5);
    transform: scale(1.05);
}

/* Headings */
h1, h2, h3 {
    color: #00BFFF;
    font-weight: 700;
}

/* Lists */
ul li {
    color: #C7D0D9;
    margin-bottom: 6px;
}

/* Upload Box Animation */
@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(20px);}
    100% {opacity: 1; transform: translateY(0);}
}

/* Footer */
.footer {
    text-align: center;
    color: #8B949E;
    font-size: 13px;
    margin-top: 40px;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =============== NAVBAR ===============
st.markdown("""
<div class='navbar'>
  <div class='brand'>🤖 AI Image Classifier</div>
  <div>
    <a href='#overview'>Overview</a>
    <a href='#upload'>Upload</a>
    <a href='#model'>Model</a>
    <a href='#about'>About</a>
  </div>
</div>
""", unsafe_allow_html=True)

# =============== HERO SECTION ===============
st.markdown("""
<div class='section' style='text-align:center; animation: fadeIn 1.5s;'>
  <h1>Welcome to the AI Image Classifier</h1>
  <p style='color:#AAB2BF;'>Upload an image and let our deep learning model predict its category with precision.</p>
</div>
""", unsafe_allow_html=True)

# =============== OVERVIEW SECTION ===============
st.markdown("""
<div id='overview' class='section' style='animation: fadeIn 1.5s;'>
  <h3>🌌 Overview</h3>
  <p>This app uses a <b>Convolutional Neural Network (CNN)</b> trained on the <b>CIFAR-10</b> dataset.</p>
  <ul>
    <li>⚡ Real-time image analysis</li>
    <li>🧠 Powered by TensorFlow</li>
    <li>📄 Auto PDF report generation</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# =============== UPLOAD SECTION ===============
st.markdown("""
<div id='upload' class='section' style='text-align:center; animation: fadeIn 1.5s;'>
  <h3 style='font-size:26px; color:#00BFFF;'>📤 Upload Image</h3>
  <p style='color:#AAB2BF; font-size:15px;'>Drag & drop your image below or click to browse<br>
  <span style='color:#00BFFF;'>Supported formats: JPG, JPEG, PNG</span></p>

  <div style="
      border: 2px dashed #00BFFF;
      border-radius: 20px;
      padding: 50px;
      margin-top: 20px;
      background: radial-gradient(circle at top left, #0B0E11, #12151B);
      box-shadow: 0 0 25px rgba(0, 191, 255, 0.05);
      transition: all 0.3s ease;
  " onmouseover="this.style.boxShadow='0 0 30px rgba(0,191,255,0.3)'"
    onmouseout="this.style.boxShadow='0 0 25px rgba(0,191,255,0.05)'">
    <p style='color:#7BA4C9; font-size:15px; margin:0;'>Drop your file here 👇</p>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if "history" not in st.session_state:
    st.session_state.history = []

# =============== PROCESS IMAGE ===============
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="📸 Uploaded Image", width=200)
    st.write("⏳ Processing...")

    # Prediction
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_name = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    st.session_state.history.append((class_name, confidence))

    # Result Section
    st.markdown(f"""
        <div class='section' style='text-align:center; animation: fadeIn 1.5s;'>
            <h3>🔍 Prediction Result</h3>
            <h2 style='color:#00FFAA;'>{class_name}</h2>
            <p>Confidence: <b>{confidence:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

    # Insight Section
    st.markdown(f"""
    <div class='section' style='animation: fadeIn 1.5s;'>
      <h3>💡 AI Insight</h3>
      <p>The model identifies key visual patterns like edges, color distribution, and object shape to classify images accurately.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model Info Section
    st.markdown("""
    <div id='model' class='section' style='animation: fadeIn 1.5s;'>
      <h3>🧠 Model Information</h3>
      <ul>
        <li>Architecture: Convolutional Neural Network (CNN)</li>
        <li>Dataset: CIFAR-10</li>
        <li>Accuracy: ~80%</li>
        <li>Framework: TensorFlow / Keras</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # PDF report
    def create_pdf():
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 20)
        c.drawString(200, height - 80, "AI Image Classification Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 110, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, height - 140, f"Predicted Class: {class_name}")
        c.drawString(50, height - 160, f"Confidence: {confidence:.2f}%")
        img = Image.open(uploaded_file)
        c.drawImage(ImageReader(img), 150, height - 400, width=200, preserveAspectRatio=True)
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    pdf_buffer = create_pdf()
    st.markdown("<div class='section' style='animation: fadeIn 1.5s;'><h3>📥 Download Report</h3>", unsafe_allow_html=True)
    st.download_button("Download PDF Report", pdf_buffer, f"{class_name}_report.pdf", "application/pdf")

# =============== ABOUT SECTION ===============
st.markdown("""
<div id='about' class='section' style='text-align:center; animation: fadeIn 1.5s;'>
  <h3>👨‍💻 About Developer</h3>
  <p>Developed by <b>Mohamed Hasham</b> — passionate about AI and computer vision.</p>
  <a href='mailto:youremail@example.com' style='color:#00BFFF;'>📧 Contact Me</a>
</div>

<div class='footer'>© 2025 | Developed by Mohamed Hasham | Powered by TensorFlow & Streamlit</div>
""", unsafe_allow_html=True)
