import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="SpamShield AI", page_icon="📧")

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    try:
        image = Image.open("Photo1.jpg")
        st.image(image, width=150)
    except:
        st.warning("Profile image not found.")

    st.markdown("<h3 style='text-align: center;'>Farhan Ahmed</h3>", unsafe_allow_html=True)

    st.markdown("### 🤝 Connect With Me")
    st.markdown("""
    - 📧 [frhanahmedf21@gmail.com](mailto:frhanahmedf21@gmail.com)
    - 💼 [LinkedIn](https://linkedin.com/in/farhanahmedf21)
    - 💻 [GitHub](https://github.com/frhanahmed)
    - 💬 [WhatsApp](https://wa.me/918910080891)
    """)

    st.markdown("### 🗂️ Source Code")
    st.markdown("[🔗 GitHub Repository](https://github.com/frhanahmed/SpamPrediction.git)")

# =============================
# MAIN TITLE
# =============================
st.title("📧 Email & SMS Spam Detection System")
st.write("Enter a message below to check whether it is Spam or Not Spam.")

# =============================
# LOAD MODEL & VECTORIZER
# =============================
ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# =============================
# USER INPUT
# =============================
input_sms = st.text_area("✉️ Enter your message here")

if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]

        st.subheader("Prediction Result:")

        if result == 1:
            confidence = round(probability[1] * 100, 2)
            st.error(f"🚨 Spam Message Detected ({confidence}% confidence)")
        else:
            confidence = round(probability[0] * 100, 2)
            st.success(f"✅ This is NOT Spam ({confidence}% confidence)")

# =============================
# CONTACT SECTION
# =============================
st.write("Feel free to send me a message using the form below!")

with st.expander("📬 Contact Me"):
    contact_form = """
        <form action="https://formsubmit.co/frhanahmedf21@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your Name" required style="width: 100%; padding: 8px;border-radius: 5px;background-color: azure;color: black;"><br><br>
        <input type="email" name="email" placeholder="Your Email" required style="width: 100%; padding: 8px;border-radius: 5px;background-color: azure;color: black;"><br><br>
        <textarea name="message" placeholder="Your message here..." rows="5" required style="width: 100%; padding: 8px;border-radius: 5px;background-color: azure;color: black;"></textarea><br><br>
        <div style="text-align: center;">
        <button type="submit" 
            style="padding: 10px 20px; border-radius: 5px; background-color: rgb(149, 68, 224); color: white;margin-bottom: 5px;">
            Send Message
        </button>
        </div>
        </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)