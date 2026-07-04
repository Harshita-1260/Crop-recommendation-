import streamlit as st
import joblib
import numpy as np

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱"
)

# -------------------- Load Model --------------------
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
.stButton>button{
    width:100%;
    background-color:#2E8B57;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#1f6f43;
    color:white;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Title --------------------
st.title("🌱 Crop Recommendation System")

st.markdown("""
### Smart Crop Prediction using Machine Learning

Enter the soil nutrient and weather details below to predict the most suitable crop.
""")

st.divider()

# -------------------- Input Fields --------------------

st.subheader("🌾 Enter Soil & Weather Details")

N = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)

P = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0)

K = st.number_input("Potassium (K)", min_value=0.0, value=43.0)

temperature = st.number_input("Temperature (°C)", value=20.8)

humidity = st.number_input("Humidity (%)", value=82.0)

ph = st.number_input("Soil pH", value=6.5)

rainfall = st.number_input("Rainfall (mm)", value=202.0)

st.divider()

# -------------------- Prediction --------------------

if st.button("🌱 Predict Crop", use_container_width=True):

    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    crop = label_encoder.inverse_transform(prediction)

    st.success(f"🌾 Recommended Crop : **{crop[0].title()}**")

    st.balloons()

    st.info(
        "This prediction is generated using a Logistic Regression Machine Learning model."
    )

    st.subheader("📋 Input Summary")

    st.table({
        "Parameter": [
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Temperature",
            "Humidity",
            "Soil pH",
            "Rainfall"
        ],
        "Value": [
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]
    })

# -------------------- Footer --------------------

st.markdown("---")

st.markdown("""
<div class="footer">

🌱 <b>Crop Recommendation System</b><br><br>

Developed by <b>Bug Creators 🐞</b><br><br>

Python • NumPy • Pandas • Scikit-Learn • Streamlit

</div>
""", unsafe_allow_html=True)