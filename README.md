# Anomali Tespiti (Unsupervised Learning) – Ağ Trafiği Analizi

Bu proje, **ağ trafiği verileri** üzerinde **etiketsiz (unsupervised)** öğrenme yaklaşımları kullanarak **anomali / saldırı tespiti** yapmayı amaçlamaktadır.

Çalışmada, özellikle **DDOS saldırılarının** normal (BENIGN) trafikten ayrıştırılması hedeflenmiştir.

---

## 🎯 Projenin Amacı

- Sadece **normal (BENIGN)** verilerle model eğitmek  
- Anormal ağ davranışlarını **önceden etiket bilgisi olmadan** tespit etmek  
- Farklı anomaly detection algoritmalarını karşılaştırmak  

---

## 📂 Veri Seti

Veri seti depoda **bulunmamaktadır**. Kod, çalışma dizininde aşağıdaki CSV dosyasını bekler:

```
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Kodun veriden beklentileri:

- Sütun isimlerindeki baştaki/sondaki boşluklar `str.strip()` ile temizlenir.
- `Label` sütunu bulunmalıdır; `BENIGN` olan kayıtlar **normal (0)**, diğerleri **anomali (1)** olarak işaretlenir. Etiketler **yalnızca değerlendirme** için kullanılır.

---

## ⚙️ Yöntem

1. **Ön işleme**
   - `Flow ID`, `Source IP`, `Destination IP`, `Timestamp` sütunları (varsa) çıkarılır.
   - Yalnızca sayısal sütunlar kullanılır.
   - `inf` / `-inf` değerleri `NaN`'a çevrilir ve `NaN` içeren satırlar silinir.
2. **Eğitim / test ayrımı**
   - Eğitim verisi: yalnızca `BENIGN` kayıtlar
   - Test verisi: tüm kayıtlar
3. **Ölçekleme:** `StandardScaler` (eğitim verisine fit edilir)
4. **Modeller**

| Model | Parametreler |
|---|---|
| Isolation Forest | `n_estimators=200`, `contamination=0.05`, `random_state=42` |
| One-Class SVM | `kernel="rbf"`, `nu=0.05`, `gamma="scale"` |
| Elliptic Envelope | `contamination=0.05`, `random_state=42` |

Modellerin `-1` (anomali) / `1` (normal) çıktıları `1` (anomali) / `0` (normal) biçimine dönüştürülür.

5. **Değerlendirme:** Her model için `classification_report` ve `confusion_matrix` konsola yazdırılır.

---

## 📊 Sonuçlar

Depoda kaydedilmiş bir çıktı bulunmadığından sayısal sonuçlar burada verilmemiştir. Sonuçlar, script çalıştırıldığında her model için konsola yazdırılır.

---

## 🗂️ Proje Yapısı

```
.
├── Untitled-1.py   # Ön işleme, üç anomali tespit modeli ve değerlendirme
└── README.md
```

---

## 🚀 Çalıştırma

```bash
pip install -r requirements.txt
# Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv dosyasını script ile aynı dizine koyun
python Untitled-1.py
```

---

## 📦 Gereksinimler

- pandas
- numpy
- scikit-learn
