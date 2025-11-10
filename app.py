import streamlit as st
import joblib
import re
import string
import nltk

# Import yang diperlukan
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 0. Konfigurasi Halaman Streamlit ---
# Mengatur layout menjadi 'wide' (lebar) untuk memberikan ruang lebih pada elemen (termasuk st.code)
st.set_page_config(page_title="Detektor Spam Bahasa Indonesia", layout="wide")

# --- 1. Muat Sumber Daya (Model, Vectorizer, NLTK, Stemmer) ---
@st.cache_resource
def load_resources():
    """
    Memuat model dan vectorizer, serta memastikan semua NLTK resources terunduh.
    """
    try:
        # Pemuatan Model dan Vectorizer
        model = joblib.load("spam_model_nb.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        
        # Pengunduhan NLTK Resources secara eksplisit (PENTING untuk deployment)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True) 
        nltk.download('punkt_tab', quiet=True) # FIX untuk LookupError punkt_tab
        
        # Inisialisasi Sumber Daya NLP
        list_stopwords = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()
        
        return model, vectorizer, list_stopwords, stemmer
    
    except FileNotFoundError:
        st.error("❌ File model atau vectorizer (.joblib) tidak ditemukan! Pastikan file berada di direktori yang sama.")
        return None, None, set(), None
    except Exception as e:
        st.error(f"❌ Gagal memuat sumber daya atau inisialisasi NLP: {e}")
        return None, None, set(), None

# Panggil fungsi untuk memuat semua sumber daya
model, vectorizer, LIST_STOPWORDS, STEMMER = load_resources()

# --- 2. Fungsi Preprocessing Teks ---
def text_preprocessing(text, stemmer, list_stopwords):
    """Melakukan langkah preprocessing yang sama seperti saat pelatihan model."""
    
    if not stemmer:
        return text 

    # Case folding dan hapus angka
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    
    # Hapus punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Filtering (Stopword removal)
    tokens = [word for word in tokens if word not in list_stopwords]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

# --- 3. Fungsi Prediksi ---
def predict_text(text, model, vectorizer, stemmer, list_stopwords):
    """Membersihkan teks, mengubahnya menjadi vektor, dan melakukan prediksi."""
    
    # 1. Preprocessing
    cleaned_text = text_preprocessing(text, stemmer, list_stopwords)
    
    # 2. Transformasi menggunakan vectorizer
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # 3. Prediksi
    prediction = model.predict(text_vectorized)
    
    return prediction[0], cleaned_text

# --- 4. Antarmuka Streamlit (Main App) ---
st.title("📧 Detektor Spam Pesan Bahasa Indonesia")
st.markdown("---")

if model is not None and vectorizer is not None:
    # Area input teks (Lebar penuh karena diletakkan langsung di main content)
    user_input = st.text_area("Masukkan Pesan di Sini:", 
                              placeholder="Ketik pesan yang ingin Anda analisis (misal: Selamat! Anda memenangkan undian berhadiah).", 
                              height=150)

    # Pembagian kolom untuk tombol, hanya col1 yang digunakan untuk tombol
    col1, col2 = st.columns([1, 4]) 
    
    # Variabel untuk menyimpan hasil prediksi agar bisa diakses di luar if st.button
    result = None
    cleaned_text = ""
    
    with col1:
        if st.button("Analisis Pesan", use_container_width=True, type="primary"):
            if user_input:
                # Lakukan prediksi
                result, cleaned_text = predict_text(user_input, model, vectorizer, STEMMER, LIST_STOPWORDS)
            else:
                st.warning("Silakan masukkan teks pesan terlebih dahulu untuk dianalisis.")

    # Tampilkan Hasil dan Expander di bawah kolom (Lebar Penuh)
    if user_input and result is not None:
        st.subheader("Hasil Prediksi:")
        
        # Hasil 1 = Spam, Hasil 0 = Ham
        if result == 1:
            st.error("🚨 SPAM 🚨", icon="🚫")
            st.markdown("Pesan ini sangat mungkin adalah **pesan spam**.")
        else:
            st.success("✅ BUKAN SPAM (HAM) ✅", icon="⭐")
            st.markdown("Pesan ini terdeteksi sebagai **pesan normal/valid**.")
        
        # Expander untuk Detail Preprocessing (Akan menggunakan lebar penuh karena di luar kolom)
        # with st.expander("Lihat Detail Preprocessing"):
        #    st.caption("Langkah-langkah yang dilakukan pada teks Anda sebelum diprediksi:")
        #   st.code(cleaned_text, language='text')

else:
    st.error("Aplikasi tidak dapat berjalan karena model atau vectorizer gagal dimuat. Cek file .joblib Anda.")



