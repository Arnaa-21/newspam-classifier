import streamlit as st
import joblib
import re

# =========================
# LOAD MODEL & VECTORIZER
# =========================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Spam Email Classifier")

st.title("📩 Spam Email Classifier")

# =========================
# INPUT BOX
# =========================
user_input = st.text_area("Enter Email Text")

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # preprocess
        clean_text = preprocess(user_input)

        # vectorize
        vector = vectorizer.transform([clean_text])

        # predict
        pred = model.predict(vector)[0]

        # probability (safe handling)
        try:
            prob = model.predict_proba(vector).max()
        except:
            prob = 0.0

        # =========================
        # FIX: HANDLE NUMERIC LABELS
        # =========================
        label_map = {
            0: "ham",
            1: "spam"
        }

        # convert prediction safely
        if isinstance(pred, (int, float)):
            label = label_map.get(int(pred), "unknown")
        else:
            label = str(pred).lower()

        # =========================
        # OUTPUT
        # =========================
        if label == "spam":
            st.error(f"🚨 Spam Detected ({prob:.2f})")
        else:
            st.success(f"✅ Ham (Safe) ({prob:.2f})")

        # =========================
        # SHOW CLEANED TEXT (Q5)
        # =========================
        st.write("### 🔍 Processed Text")
        st.write(clean_text)

        # DEBUG (optional, remove later)
        # st.write("Raw Prediction:", pred)