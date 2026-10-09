# ===============================================
#   ANOMALİ TESPİTİ (UNSUPERVISED)
#   Modeller:
#      - Isolation Forest
#      - One-Class SVM
#      - Elliptic Envelope
# ===============================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------
# 1. VERİYİ OKU
# ---------------------------------------------------

df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# 👇 SÜTUN İSİMLERİNDEKİ BOŞLUKLARI SİL
df.columns = df.columns.str.strip()

print("Sütunlar:", df.columns.tolist()[-5:])  # kontrol için son 5 sütuna bak

# ---------------------------------------------------
# 2. ÖN İŞLEME
# ---------------------------------------------------

drop_cols = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Sadece sayısal kolonlar
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Label'ı binary hale getir (sadece değerlendirme için!)
df["Label_binary"] = (df["Label"] != "BENIGN").astype(int)

# Eğitim verisi: SADECE BENIGN (normal) kayıtlar
train_df = df[df["Label_binary"] == 0][numeric_cols]

# Test verisi: tüm kayıtlar
test_df = df[numeric_cols]
test_labels = df["Label_binary"]   # sadece değerlendirme amaçlı

# NaN ve sonsuzları temizle
train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()
test_labels = test_labels.loc[test_df.index]   # etiketleri temizlenen satırlarla hizala

# ---------------------------------------------------
# 3. SCALE
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(train_df)
X_test = scaler.transform(test_df)

print("Eğitim BENIGN kayıt sayısı:", X_train.shape)
print("Toplam test kayıt sayısı:", X_test.shape)

# ===================================================
# MODEL A — ISOLATION FOREST
# ===================================================

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # anomalinin oranı (tahmini)
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train)
pred_iso = iso.predict(X_test)

# ISO: normal→1, anomali→-1 → binary formatla (0 normal, 1 anomali)
pred_iso_bin = np.where(pred_iso == -1, 1, 0)

print("\n=== Isolation Forest Sonuçları ===")
print(classification_report(test_labels, pred_iso_bin, target_names=["Normal","Anomali"]))
print(confusion_matrix(test_labels, pred_iso_bin))


# ===================================================
# MODEL B — ONE-CLASS SVM
# ===================================================

oc_svm = OneClassSVM(
    kernel="rbf",
    nu=0.05,
    gamma="scale"
)

oc_svm.fit(X_train)
pred_svm = oc_svm.predict(X_test)

# OCSVM: normal→1, anomali→-1
pred_svm_bin = np.where(pred_svm == -1, 1, 0)

print("\n=== One-Class SVM Sonuçları ===")
print(classification_report(test_labels, pred_svm_bin, target_names=["Normal","Anomali"]))
print(confusion_matrix(test_labels, pred_svm_bin))


# ===================================================
# MODEL C — ELLIPTIC ENVELOPE
# ===================================================

ell = EllipticEnvelope(contamination=0.05, random_state=42)
ell.fit(X_train)
pred_ell = ell.predict(X_test)

pred_ell_bin = np.where(pred_ell == -1, 1, 0)

print("\n=== Elliptic Envelope Sonuçları ===")
print(classification_report(test_labels, pred_ell_bin, target_names=["Normal","Anomali"]))
print(confusion_matrix(test_labels, pred_ell_bin))

print("\nTüm anomaly detection modelleri tamamlandı.")
