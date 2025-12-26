import streamlit as st
import numpy as np
import time
from PIL import Image
import joblib 
import pandas as pd 
import plotly.graph_objects as go
import country_converter as coco
from map import make_diabetes_map

# PATH MODEL
MODEL_GABUNGAN_PATH = 'model_prediksi_diabetes_gabunganv1.joblib' 
MODEL_NON_LAB_PATH = 'model_prediksi_diabetes_nonlabv1.joblib' 

OPTIMAL_THRESHOLD_GABUNGAN = 0.60 

# FUNGSI PEMUATAN MODEL + FITUR
@st.cache_data
def load_and_cache_image(image_path):
    try:
        img = Image.open(image_path)
        return img
    except FileNotFoundError:
        return None

@st.cache_resource
def load_ai_models():
    time.sleep(1)
    model_nl = joblib.load(MODEL_NON_LAB_PATH)
    model_gab = joblib.load(MODEL_GABUNGAN_PATH)
    features_nl = [str(f) for f in model_nl.feature_names_in_]
    features_gab = [str(f) for f in model_gab.feature_names_in_]
    return model_nl, model_gab, features_nl, features_gab

try:
    MODEL_NL, MODEL_GAB, FEATURE_LIST_NON_LAB, FEATURE_LIST_GABUNGAN = load_ai_models()
    CACHED_IMAGE = load_and_cache_image('diabetes.jpg')
    AI_MODELS_LOADED = True
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .joblib ada. Error: {e}")
    AI_MODELS_LOADED = False
    MODEL_NL, MODEL_GAB = None, None
    FEATURE_LIST_NON_LAB = []
    FEATURE_LIST_GABUNGAN = []

# STATE MANAGEMENT
if 'step' not in st.session_state:
    st.session_state.step = 1 
if 'data_collected' not in st.session_state:
    st.session_state.data_collected = {}

def go_to_step(target_step):
    st.session_state.step = target_step

def predict_ai(features_dict, model, feature_list):
    if model is None or not hasattr(model, 'predict_proba'):
        return 0.5 
    try:
        input_values = [features_dict.get(feat) for feat in feature_list]
        if None in input_values:
            return 0.5
        df_input = pd.DataFrame([input_values], columns=feature_list)
        prob = model.predict_proba(df_input)[0, 1] 
        return prob
    except Exception as e:
        return 0.5

# STYLING
st.set_page_config(page_title="DiaLens App", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .centered-text { text-align: center; margin-bottom: 20px; }
    .big-title { font-size: 2.5em; font-weight: bold; color: #DDDDDD; }
    .subtitle { font-size: 1.5em; font-weight: 500; color: #BBBBBB; margin-top: 15px; } 
    .instruction-text { font-size: 1em; color: #AAAAAA; margin-bottom: 40px; }
    .footer-text { text-align: center; margin-top: 50px; color: #6c757d; }
    .stButton>button {
        width: 100%; height: 50px; font-size: 1.2em; font-weight: bold; border-radius: 8px;
        background-color: #38a745; color: white; border: none;
    }
    .stButton>button:focus:not(.st-ex):nth-child(1) { background-color: #6c757d; color: white; }
    .step-header { font-size: 1.8em; font-weight: bold; color: #DDDDDD;}
    </style>
""", unsafe_allow_html=True)

# STEP 1: WELCOME
def display_step_1():
    st.markdown(
        """
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .welcome-title {
            font-size: 2.8em;
            font-weight: 700;
            color: #E76F51;
            text-align: center;
            margin: 20px 0 10px;
            animation: fadeIn 0.8s ease-out;
        }
        .welcome-subtitle {
            font-size: 1.4em;
            color: #2A9D8F;
            text-align: center;
            margin-bottom: 30px;
            animation: fadeIn 0.8s ease-out 0.2s both;
        }
        .welcome-instruction {
            font-size: 1.1em;
            color: #6C757D;
            text-align: center;
            max-width: 600px;
            margin: 0 auto 40px;
            line-height: 1.6;
            animation: fadeIn 0.8s ease-out 0.4s both;
        }
        .footer-step {
            text-align: center;
            margin-top: 30px;
            color: #ADB5BD;
            font-size: 0.9em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="welcome-title">🩺 Selamat Datang di DiaLens!</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-subtitle">AI untuk Deteksi Risiko Diabetes yang Cerdas & Personal</div>', unsafe_allow_html=True)
    
    col_img_left, col_img_center, col_img_right = st.columns([1, 5, 1])
    with col_img_center:
        if CACHED_IMAGE:
            st.image(CACHED_IMAGE, use_container_width=True, caption="Ilustrasi: Diabetes Awareness")
        else:
            st.markdown(
                '<div style="background:#f8f9fa; height:200px; display:flex; align-items:center; justify-content:center; border-radius:10px; color:#6c757d;">Gambar tidak tersedia</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="welcome-instruction">Lengkapi profil kesehatan Anda dalam 2 menit, dan dapatkan prediksi risiko diabetes berbasis AI — dengan atau tanpa data laboratorium.</div>',
        unsafe_allow_html=True
    )

    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
    with col_btn_center:
        if st.button("✨ Mulai Sekarang", key="btn_mulai", use_container_width=True):
            go_to_step(2)

    st.markdown('<div class="footer-step">Langkah 1 dari 3</div>', unsafe_allow_html=True)

# STEP 2: INPUT DATA
def display_step_2():
    st.markdown(
        """
        <style>
        .section-header {
            font-size: 1.6em;
            font-weight: 600;
            color: #E76F51;
            margin: 1.8rem 0 1rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid #f0f0f0;
        }
        .input-hint {
            font-size: 0.85em;
            color: #6c757d;
            margin-top: -8px;
            margin-bottom: 12px;
        }
        .required::after {
            content: " *";
            color: #E76F51;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="step-header">📝 Profil Kesehatan Pribadi</div>', unsafe_allow_html=True)
    st.markdown("Lengkapi data di bawah ini untuk mendapatkan prediksi risiko diabetes yang akurat.")

    if not FEATURE_LIST_NON_LAB or not AI_MODELS_LOADED:
        st.error("⚠️ Model atau fitur tidak tersedia. Periksa file .joblib.")
        return

    #BAGIAN 1: DATA NON-LAB (WAJIB)
    st.markdown('<div class="section-header required">Data Dasar & Gaya Hidup (Wajib)</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.data_collected['Age'] = st.number_input(
                "Usia (Tahun)",
                min_value=20,
                max_value=90,
                value=35,
                step=1,
                key="age"
            )
            st.caption("Rentang usia: 20–90 tahun")

            st.session_state.data_collected['DietQuality'] = st.slider(
                "Kualitas Diet",
                0,
                10,
                5,
                key="diet",
                help="0 = Pola makan tidak sehat, 10 = Sangat seimbang (sayur, buah, serat tinggi, gula rendah)"
            )
            st.markdown('<div class="input-hint">Skor 7+ umumnya terkait risiko lebih rendah</div>', unsafe_allow_html=True)

        with col2:
            st.session_state.data_collected['HealthLiteracy'] = st.slider(
                "Literasi Kesehatan",
                0,
                10,
                5,
                key="literacy",
                help="Seberapa paham Anda tentang informasi kesehatan, pencegahan, dan pengelolaan penyakit"
            )
            st.markdown('<div class="input-hint">Pemahaman kesehatan memengaruhi keputusan pencegahan</div>', unsafe_allow_html=True)

            st.session_state.data_collected['Smoking'] = st.radio(
                "Apakah Anda merokok?",
                [0, 1],
                format_func=lambda x: "❌ Tidak" if x == 0 else "✅ Ya",
                horizontal=True,
                key="smoking"
            )


    #BAGIAN 2: RIWAYAT & GEJALA
    st.markdown('<div class="section-header required">Riwayat Kesehatan & Gejala</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        col_left, col_right = st.columns(2)

        with col_left:
            st.session_state.data_collected['Hypertension'] = st.radio(
                "Apakah Anda memiliki hipertensi (tekanan darah tinggi)?",
                [0, 1],
                format_func=lambda x: "❌ Tidak" if x == 0 else "✅ Ya",
                horizontal=True,
                key="hyper"
            )
            st.session_state.data_collected['FamilyHistoryDiabetes'] = st.radio(
                "Apakah ada anggota keluarga dekat (orang tua/saudara) dengan diabetes?",
                [0, 1],
                format_func=lambda x: "❌ Tidak" if x == 0 else "✅ Ya",
                horizontal=True,
                key="family"
            )

        with col_right:
            st.markdown("**Gejala Klinis (centang jika pernah mengalami):**")
            st.session_state.data_collected['FrequentUrination'] = st.checkbox(
                "Sering buang air kecil (terutama malam hari)", key="urine"
            )
            st.session_state.data_collected['ExcessiveThirst'] = st.checkbox(
                "Rasa haus berlebihan yang tidak biasa", key="thirst"
            )
            st.session_state.data_collected['UnexplainedWeightLoss'] = st.checkbox(
                "Penurunan berat badan tanpa diet atau olahraga", key="weight"
            )

            # Konversi checkbox ke 0/1
            for key in ['FrequentUrination', 'ExcessiveThirst', 'UnexplainedWeightLoss']:
                st.session_state.data_collected[key] = int(st.session_state.data_collected[key])

    # BAGIAN 3: DATA LAB 
    st.markdown('<div class="section-header">Data Laboratorium (Opsional, Tapi Sangat Direkomendasikan)</div>', unsafe_allow_html=True)
    st.info("💡 Jika Anda pernah melakukan pemeriksaan gula darah, masukkan nilainya. Jika tidak, biarkan **0** — sistem akan gunakan nilai normal.")

    with st.container(border=True):
        lab_col1, lab_col2 = st.columns(2)

        with lab_col1:
            hba1c_input = st.text_input(
                "HbA1c (%) – Rata-rata gula darah 2–3 bulan terakhir",
                value="0",
                help="Normal: 4.0–5.6% | Prediabetes: 5.7–6.4% | Diabetes: ≥6.5%",
                key="hba1c_input"
            )
            try:
                hba1c_val = float(hba1c_input.strip())
                hba1c_val = hba1c_val if hba1c_val != 0 and 4.0 <= hba1c_val <= 10.0 else 0.0
            except ValueError:
                hba1c_val = 0.0
                st.warning("⚠️ Format HbA1c tidak valid. Gunakan angka desimal seperti 5.7 atau 0.")
            st.session_state.data_collected['HbA1c'] = hba1c_val

        with lab_col2:
            gdp_input = st.text_input(
                "Gula Darah Puasa (mg/dL)",
                value="0",
                help="Normal: 70–99 | Prediabetes: 100–125 | Diabetes: ≥126",
                key="gdp_input"
            )
            try:
                gdp_val = float(gdp_input.strip())
                gdp_val = gdp_val if gdp_val != 0 and 70 <= gdp_val <= 200 else 0.0
            except ValueError:
                gdp_val = 0.0
                st.warning("⚠️ Format gula darah tidak valid. Contoh: 95 atau 0.")
            st.session_state.data_collected['FastingBloodSugar'] = gdp_val

    # NAVIGASI
    st.markdown("<br>", unsafe_allow_html=True)
    nav1, nav2 = st.columns([1, 1])

    with nav1:
        if st.button("← Kembali ke Beranda", use_container_width=True, key="back2"):
            go_to_step(1)

    with nav2:
        # Validasi kelengkapan data non-lab
        required_keys = ['Age', 'DietQuality', 'HealthLiteracy', 'Smoking', 'Hypertension',
                         'FamilyHistoryDiabetes', 'FrequentUrination', 'ExcessiveThirst',
                         'UnexplainedWeightLoss']
        incomplete = [k for k in required_keys if st.session_state.data_collected.get(k) is None]

        if incomplete:
            st.error(f"❌ Mohon lengkapi semua data wajib: {', '.join(incomplete)}")
        else:
            if st.button("➡️ Lihat Hasil Prediksi", use_container_width=True, key="next2", type="primary"):
                go_to_step(3)

    st.markdown('<div class="footer-text">Langkah 2 dari 3 • Semua data disimpan hanya di perangkat Anda</div>', unsafe_allow_html=True)

# STEP 3: HASIL PREDIKSI + KONTEKS GLOBAL

def display_step_3():
    st.markdown(
        """
        <style>
        .risk-high { color: #E76F51; font-weight: bold; }
        .risk-medium { color: #F4A261; font-weight: bold; }
        .risk-low { color: #2A9D8F; font-weight: bold; }
        .recommendation-box {
            padding: 16px;
            border-radius: 10px;
            margin: 16px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="step-header">📊 Hasil Prediksi Risiko Diabetes</div>', unsafe_allow_html=True)
    st.markdown("---")
    data = st.session_state.data_collected

    # --- MODEL NON-LAB (SELALU DITAMPILKAN) ---
    input_nl = {f: data[f] for f in FEATURE_LIST_NON_LAB}
    prob_nl = predict_ai(input_nl, MODEL_NL, FEATURE_LIST_NON_LAB)
    risk_nl = "Tinggi" if prob_nl >= OPTIMAL_THRESHOLD_GABUNGAN else ("Sedang" if prob_nl >= 0.5 else "Rendah")

    st.subheader("🧠 Prediksi Berbasis Gaya Hidup & Riwayat (Tanpa Lab)")
    with st.container(border=True):
        risk_color = "risk-high" if risk_nl == "Tinggi" else ("risk-medium" if risk_nl == "Sedang" else "risk-low")
        st.markdown(f"**Tingkat Risiko**: <span class='{risk_color}'>{risk_nl}</span>", unsafe_allow_html=True)
        st.markdown(f"**Probabilitas**: {prob_nl:.1%}")
        st.progress(float(prob_nl))

    # --- CEK DATA LAB ---
    hba1c_raw = data.get('HbA1c', 0)
    gdp_raw = data.get('FastingBloodSugar', 0)
    lab_available = (hba1c_raw != 0) or (gdp_raw != 0)

    prob_gab = None
    risk_gab = None

    if lab_available:
        input_gab = {}
        for feat in FEATURE_LIST_GABUNGAN:
            val = data.get(feat)
            if feat == 'HbA1c' and val == 0:
                val = 5.5
            elif feat == 'FastingBloodSugar' and val == 0:
                val = 95.0
            input_gab[feat] = val

        prob_gab = predict_ai(input_gab, MODEL_GAB, FEATURE_LIST_GABUNGAN)
        risk_gab = "Tinggi" if prob_gab >= OPTIMAL_THRESHOLD_GABUNGAN else ("Sedang" if prob_gab >= 0.5 else "Rendah")

        st.subheader("🧪 Prediksi dengan Data Laboratorium")
        if hba1c_raw == 0 or gdp_raw == 0:
            st.warning("⚠️ Sebagian data lab diasumsikan normal karena tidak diisi.")
        else:
            st.success("✅ Prediksi diperkuat dengan hasil laboratorium Anda!")

        with st.container(border=True):
            risk_color = "risk-high" if risk_gab == "Tinggi" else ("risk-medium" if risk_gab == "Sedang" else "risk-low")
            st.markdown(f"**Tingkat Risiko**: <span class='{risk_color}'>{risk_gab}</span>", unsafe_allow_html=True)
            st.markdown(f"**Probabilitas**: {prob_gab:.1%}")
            st.progress(float(prob_gab))

    #  SARAN PERSONALISASI BERDASARKAN RISIKO TERTINGGI
    final_risk = risk_gab if risk_gab else risk_nl
    st.header("💡 Rekomendasi Personal dari DiaLens")

    if final_risk == "Rendah":
        st.markdown(
            """
            <div class="recommendation-box" style="background-color: #e8f5e9; border-left: 4px solid #2A9D8F; color: #264653;">
            <h4 style="color: #2A9D8F; margin-top: 0;">🟢 Risiko Rendah – Pertahankan Gaya Hidup Sehat!</h4>
            <ul>
                <li>Lanjutkan pola makan seimbang dan aktivitas fisik rutin (≥150 menit/minggu).</li>
                <li>Periksa gula darah setiap 2–3 tahun, terutama jika usia &gt;40 tahun.</li>
                <li>Pertahankan literasi kesehatan — Anda sudah di jalur yang tepat!</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif final_risk == "Sedang":
        st.markdown(
            """
            <div class="recommendation-box" style="background-color: #fff8e1; border-left: 4px solid #F4A261; color: #264653;">
            <h4 style="color: #E9C46A; margin-top: 0;">🟡 Risiko Sedang – Waspadai & Ambil Langkah Pencegahan</h4>
            <ul>
                <li>Kurangi konsumsi gula, minuman manis, dan makanan olahan.</li>
                <li>Tingkatkan aktivitas fisik (jalan cepat, bersepeda, olahraga ringan).</li>
                <li>Lakukan pemeriksaan gula darah puasa atau HbA1c dalam 3–6 bulan.</li>
                <li>Konsultasi dengan tenaga kesehatan untuk skrining lebih lanjut.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:  # Tinggi
        st.markdown(
            """
            <div class="recommendation-box" style="background-color: #ffebee; border-left: 4px solid #E76F51; color: #1D3557;">
            <h4 style="color: #E76F51; margin-top: 0;">🔴 Risiko Tinggi – Segera Konsultasi Medis!</h4>
            <ul>
                <li>Anda berada dalam kelompok berisiko tinggi untuk diabetes tipe 2.</li>
                <li><strong>Segera konsultasi dengan dokter</strong> untuk pemeriksaan lengkap (HbA1c, GDP, profil lipid).</li>
                <li>Hindari gula tambahan, rokok, dan gaya hidup sedentari.</li>
                <li>Pertimbangkan program pencegahan diabetes terstruktur (jika tersedia).</li>
            </ul>
            <p style="font-style: italic; margin-top: 10px; color: #555;">
                Ingat: Diagnosis dini dan intervensi gaya hidup dapat menurunkan risiko hingga 58%.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 🌍 KONTEKS GLOBAL (DATA IDF 2024)
    st.header("🌍 Fakta Global: Diabetes di Dunia (IDF Atlas 2024)")
    st.markdown("""
    Berdasarkan data resmi dari **International Diabetes Federation (IDF)**:
    
    - **589 juta** orang dewasa (20–79 tahun) hidup dengan diabetes → **1 dari 9 orang**.  
    - Diprediksi meningkat menjadi **853 juta pada 2050** → **1 dari 8 orang**.  
    - **3,4 juta** kematian akibat diabetes terjadi pada 2024 → **1 kematian setiap 9 detik**.  
    - **252 juta (43%)** kasus diabetes **tidak terdiagnosis** — 81% di antaranya di negara berpenghasilan rendah-menengah.  
    - Belanja kesehatan global untuk diabetes: **USD 1,015 triliun** pada 2024.  
    - **635 juta** orang memiliki **toleransi glukosa terganggu (IGT)**, dan **488 juta** memiliki **glukosa puasa terganggu (IFG)** — kondisi pradiabetes.  
    - **1 dari 5 kelahiran hidup** dipengaruhi oleh hiperglikemia dalam kehamilan.
    
    Sumber: [IDF Diabetes Atlas](https://diabetesatlas.org/data/en/world/)
    """)

    # 🗺️ PETA GLOBAL
    world_map = make_diabetes_map()
    if world_map is not None:
        st.plotly_chart(world_map, use_container_width=True)

    # NAVIGASI
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⇦ Kembali ke Input Data", key="back3", use_container_width=True):
        go_to_step(2)

    st.markdown('<div class="footer-text">Langkah 3 dari 3 • Hasil ini bukan diagnosis medis</div>', unsafe_allow_html=True)

# JALANKAN APLIKASI
if not AI_MODELS_LOADED:
    st.stop()

if st.session_state.step == 1:
    display_step_1()
elif st.session_state.step == 2:
    display_step_2()
elif st.session_state.step == 3:
    display_step_3()