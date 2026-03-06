import streamlit as st
import joblib
import pandas as pd

# Load preprocess dan model (Pastiin filenya ada di folder yang sama)
# Dosen lu simpannya di folder 'artifacts/', kalau kita kemaren simpannya di folder utama
scaler = joblib.load("preprocessing.pkl")
model = joblib.load("model.pkl")

def main():
    st.title('Machine Learning Heart Attack Prediction')
    st.write('Masukkan data medis pasien untuk melihat hasil prediksi.')

    # Bikin 2 kolom biar UI-nya gak kepanjangan ke bawah
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Numerik")
        age = st.number_input('Umur (age)', min_value=1, max_value=120, value=50)
        trestbps = st.number_input('Tekanan Darah (trestbps)', min_value=50, max_value=250, value=130)
        chol = st.number_input('Kolesterol (chol)', min_value=100, max_value=600, value=240)
        thalach = st.number_input('Detak Jantung Maks (thalach)', min_value=60, max_value=250, value=150)
        oldpeak = st.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)

    with col2:
        st.subheader("Data Kategorik")
        # Karena pas training kita ubah jadi string, di sini kita set value-nya sebagai string
        sex = st.selectbox('Jenis Kelamin (sex)', ['0', '1'], help='1 = Laki-laki, 0 = Perempuan')
        cp = st.selectbox('Tipe Nyeri Dada (cp)', ['0', '1', '2', '3'])
        fbs = st.selectbox('Gula Darah Puasa > 120 (fbs)', ['0', '1'])
        restecg = st.selectbox('Hasil EKG (restecg)', ['0', '1', '2'])
        exang = st.selectbox('Angina (exang)', ['0', '1'])
        slope = st.selectbox('Kemiringan ST (slope)', ['0', '1', '2'])
        ca = st.selectbox('Jumlah Pembuluh Darah (ca)', ['0', '1', '2', '3', '4'])
        thal = st.selectbox('Thalassemia (thal)', ['0', '1', '2', '3'])

    if st.button('Make Prediction'):
        # Bungkus semua inputan ke dalam dictionary
        features = {
            'age': [age], 'trestbps': [trestbps], 'chol': [chol], 
            'thalach': [thalach], 'oldpeak': [oldpeak],
            'sex': [sex], 'cp': [cp], 'fbs': [fbs], 'restecg': [restecg], 
            'exang': [exang], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        }
        
        result = make_prediction(features)
        
        st.markdown("---")
        if result == 1:
            st.error('The prediction is: 1 (High Risk / Risiko Tinggi)')
        else:
            st.success('The prediction is: 0 (Low Risk / Risiko Rendah)')

def make_prediction(features):
    # Ubah ke DataFrame biar nama kolomnya kebaca sama ColumnTransformer
    input_df = pd.DataFrame(features)
    
    # Scale & Encode (pakai variabel 'scaler' alias preprocessor kita)
    X_processed = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(X_processed)
    return prediction[0]

if __name__ == '__main__':
    main()