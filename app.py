import streamlit as app
import joblib
import re
import string
import nltk

# Import yang diperlukan
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 0. Konfigurasi Halaman Streamlit ---
app.set_page_config(
    page_title="Detektor Teks Spam Bahasa Indonesia",
    page_icon="https://plai.ac.id/assets/images/logo/MAIN-min.png",
    layout="wide"
)

# --- 1. Muat Sumber Daya (Model, Vectorizer, NLTK, Stemmer) ---
@app.cache_resource
def load_resources():
    """
    Memuat model dan vectorizer, serta memastikan semua NLTK resources terunduh.
    """
    try:
        # Pemuatan Model dan Vectorizer
        model = joblib.load("spam_pesan_model_svm.joblib")
        vectorizer = joblib.load("tfidf_vectorizer_spam_svm.joblib")
        
        # Pengunduhan NLTK Resources secara eksplisit (PENTING untuk deployment)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True) 
        nltk.download('punkt_tab', quiet=True) # FIX untuk LookupError punkt_tab
        
        # Inisialisasi Sumber Daya NLP
        list_stopwords = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()
        
        return model, vectorizer, list_stopwords, stemmer
    
    except FileNotFoundError:
        app.error("❌ File model atau vectorizer (.joblib) tidak ditemukan! Pastikan file berada di direktori yang sama.")
        return None, None, set(), None
    except Exception as e:
        app.error(f"❌ Gagal memuat sumber daya atau inisialisasi NLP: {e}")
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
col1, col2, col3 = app.columns(3)

with col2:
    app.image(
        "https://plai.ac.id/assets/images/logo/BMD-LOGO.webp",
        width=900
    )

app.title(
    "Detektor Teks Spam Bahasa Indonesia",
    text_alignment="center"
)
app.markdown("---")

if model is not None and vectorizer is not None:
    # Area input teks (Lebar penuh karena diletakkan langsung di main content)
    user_input = app.text_area("Masukkan Pesan di Sini:", 
                              placeholder="Ketik pesan yang ingin Anda analisis (misal: Selamat! Anda memenangkan undian berhadiah).", 
                              height=250)

    # Pembagian kolom untuk tombol, hanya col2 yang digunakan untuk tombol
    col1, col2, col3 = app.columns([2, 1, 2]) 
    
    # Variabel untuk menyimpan hasil prediksi agar bisa diakses di luar if app.button
    result = None
    cleaned_text = ""
    
    with col2:
        if app.button("Analisis Pesan", use_container_width=True, type="primary"):
            if user_input:
                result, cleaned_text = predict_text(user_input, model, vectorizer, STEMMER, LIST_STOPWORDS)
            else:
                app.warning("Silakan masukkan teks pesan terlebih dahulu untuk dianalisis.")

    # Tampilkan Hasil dan Expander di bawah kolom (Lebar Penuh)
    if user_input and result is not None:
        app.subheader("Hasil Prediksi:")
        
        # Hasil 1 = Spam, Hasil 0 = Ham
        if result == 1:
            app.error("🚨 SPAM 🚨", icon="🚫")
            app.markdown("Pesan ini sangat mungkin adalah **pesan spam**.")
        else:
            app.success("✅ BUKAN SPAM (HAM) ✅", icon="⭐")
            app.markdown("Pesan ini terdeteksi sebagai **pesan normal/non spam**.")
        
        # Expander untuk Detail Preprocessing (Akan menggunakan lebar penuh karena di luar kolom)
        with app.expander("Lihat Detail Preprocessing"):
            app.caption("Langkah-langkah yang dilakukan pada teks Anda sebelum diprediksi:")
            app.code(cleaned_text, language='text')

else:
    app.error("Aplikasi tidak dapat berjalan karena model atau vectorizer gagal dimuat. Cek file .joblib Anda.")










