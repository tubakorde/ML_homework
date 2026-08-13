# Kredi Riski Tahmini — Uçtan Uca Makine Öğrenmesi Projesi

**Türkiye Yapay Zekâ Akademisi — Makine Öğrenmesi Final Ödevi**

Bir bankaya kredi başvurusunda bulunan müşterinin "iyi" mi yoksa "riskli" mi olduğunu,
başvuru anında bilinen bilgilerle önceden tahmin eden uçtan uca bir sınıflandırma projesi.

---

## 1. Projenin Amacı

Bir banka, kredi başvurusunu değerlendirirken tek bir soruya cevap aramak zorunda:
**"Bu müşteri krediyi geri öder mi?"**

Bu proje, geçmişte sonuçlanmış 1.000 gerçek başvurunun verisini kullanarak bu kararı
veri temelli hale getirmeyi amaçlıyor. Derste işlenen makine öğrenmesi akışının
tamamı uygulanıyor: veri inceleme → ön işleme → öznitelik mühendisliği → model eğitimi
→ çapraz doğrulama → hiperparametre ayarlama → değerlendirme → yorumlama.

### Neden bu problem?

Bu veri setinin sahibi, hataların maliyetinin **eşit olmadığını** açıkça belirtmiş:

| Hata türü | Anlamı | Maliyet |
|---|---|---|
| **FN** — riskliye "iyi" demek | Batak krediyi onayladık | **5 birim** |
| **FP** — iyiye "riskli" demek | İyi müşteriyi geri çevirdik | **1 birim** |

Bu asimetri, projenin metrik seçimini baştan sona belirliyor. Başarı ölçütümüz
`accuracy` değil; **riskli sınıfın yakalanma oranı (recall)**, **ROC-AUC** ve
**toplam maliyet**.

---

## 2. Veri Seti

**UCI Machine Learning Repository — Statlog (German Credit Data)**
Prof. Dr. Hans Hofmann, Universität Hamburg.

| Özellik | Değer |
|---|---|
| Satır sayısı | 1.000 başvuru |
| Değişken sayısı | 20 (13 kategorik + 7 sayısal) |
| Hedef değişken | `risk` → 0 = iyi müşteri, 1 = riskli müşteri |
| Sınıf dağılımı | 700 iyi (%70) / 300 riskli (%30) — dengesiz |
| Eksik değer | Yok |
| Problem türü | **İkili sınıflandırma** (binary classification) |

Veri dosyası `data/german_credit.csv` yolunda repoda mevcuttur. Silinmiş olsa bile
betik veriyi `ucimlrepo` üzerinden UCI sunucusundan otomatik indirip aynı yere kaydeder.

### Değişkenler

Orijinal veri setinde sütunlar `Attribute1 … Attribute20` şeklinde adlandırılmış ve
kategorik değerler `A11`, `A34` gibi kodlarla kodlanmış. Betik hem sütun adlarını
hem de kategori kodlarını okunabilir Türkçe karşılıklarına çeviriyor:

`hesap_durumu`, `vade_ay`, `kredi_gecmisi`, `kredi_amaci`, `kredi_tutari`, `birikim`,
`calisma_suresi`, `taksit_orani`, `medeni_durum_cinsiyet`, `kefil_durumu`,
`ikamet_suresi`, `mulk`, `yas`, `diger_taksitler`, `konut_durumu`,
`bankadaki_kredi_sayisi`, `meslek`, `bakmakla_yukumlu_kisi`, `telefon`, `yabanci_isci`

---

## 3. Nasıl Çalıştırılır?

```bash
# 1) Depoyu klonlayın
git clone https://github.com/tubakorde/ML_homework.git
cd ML_homework

# 2) Sanal ortam oluşturun (önerilir)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 3) Bağımlılıkları kurun
pip install -r requirements.txt

# 4) Projeyi çalıştırın
python kredi_riski_tahmini.py
```

**Gereksinimler:** Python 3.9+
**Çalışma süresi:** Ortalama bir dizüstü bilgisayarda ~1-2 dakika.
**Çıktı:** Tüm adımlar konsola yazdırılır; grafikler `gorseller/` klasörüne kaydedilir.

---

## 4. Proje Yapısı

```
.
├── kredi_riski_tahmini.py     # Ana betik — tüm akış burada
├── data/
│   └── german_credit.csv      # Veri seti (1.000 satır)
├── gorseller/                 # Betiğin ürettiği grafikler
│   ├── 01_hedef_dagilimi.png
│   ├── 02_kategorik_risk.png
│   ├── 03_aykiri_degerler.png
│   ├── 04_oznitelik_onemi.png
│   ├── 05_model_karsilastirma.png
│   ├── 06_test_sonuclari.png
│   └── 07_aciklanabilirlik.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. İzlenen Adımlar

| # | Adım | Ne yapıldı? |
|---|---|---|
| 1 | Veri yükleme | CSV okundu, sütun ve kategori kodları Türkçeleştirildi |
| 2 | Hedef belirleme | `risk` (0/1), ikili sınıflandırma |
| 3 | Veri inceleme | Boyut, tipler, `describe()`, kategori bazlı risk oranları |
| 4 | Eksik değer | Teknik eksik yok; "gizli eksik" kategoriler tespit edildi |
| 5 | Aykırı değer | IQR ile incelendi, **silinmedi** (gerekçesi aşağıda) |
| 6 | Öznitelik mühendisliği | **5 yeni değişken** üretildi |
| 7 | Encoding | One-hot encoding (`drop_first=True`) → 25 sütun 55 sütuna çıktı |
| 8 | Veri ayrımı | Train %60 / Validation %20 / Test %20, `stratify=y` |
| 9 | Ölçekleme | `StandardScaler`, **yalnızca train üzerinde fit** |
| 10 | Öznitelik seçimi | Düşük varyans elemesi + korelasyon + RF önem → 55 → 42 sütun |
| 11 | Model eğitimi | **5 model**: Lojistik Regresyon, KNN, Karar Ağacı, Rastgele Orman, SVM |
| 12 | Karşılaştırma | Validation metrikleri + 5 katlı `StratifiedKFold` çapraz doğrulama |
| 13 | Hiperparametre | `GridSearchCV`, 36 kombinasyon, `scoring="roc_auc"` |
| 14 | Test | Confusion matrix, accuracy, precision, recall, F1, ROC-AUC, maliyet |
| 15 | Etik kontrol | Cinsiyet değişkeni olmadan model yeniden eğitildi |
| 16 | Açıklanabilirlik | Permütasyon önemi + lojistik regresyon katsayıları + örnek açıklama |

### Üretilen öznitelikler

| Öznitelik | Formül | Gerekçe |
|---|---|---|
| `aylik_taksit` | `kredi_tutari / vade_ay` | Aynı tutarın 12 ay ve 48 ay vadesi farklı risk taşır |
| `kredi_yas_orani` | `kredi_tutari / yas` | Genç müşterinin aynı yükü üstlenmesi daha riskli |
| `log_kredi_tutari` | `log1p(kredi_tutari)` | Sağa çarpık dağılımı simetrikleştirir |
| `toplam_yuk` | `taksit_orani × vade_ay` | Gelirin ne kadarı, ne kadar süreyle bağlanıyor? |
| `yas_grubu` | `pd.cut` ile 4 grup | Yaşın etkisi doğrusal değil |

---

## 6. Sonuçlar

### Validation karşılaştırması (%)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV AUC (ort ± std) |
|---|---|---|---|---|---|---|
| Lojistik Regresyon | 65.5 | 45.2 | **70.0** | 54.9 | 72.9 | 79.0 ± 5.2 |
| KNN | **71.5** | 57.1 | 20.0 | 29.6 | 70.1 | 72.6 ± 5.8 |
| Karar Ağacı | 60.5 | 39.6 | 60.0 | 47.7 | 63.9 | 73.9 ± **3.5** |
| Rastgele Orman | 65.0 | 43.4 | 55.0 | 48.5 | 74.1 | **80.3** ± 6.6 |
| **SVM (RBF)** | 66.5 | 45.9 | 65.0 | 53.8 | **74.4** | 80.0 ± 5.7 |

### Seçilen model: SVM (RBF), GridSearchCV ile ayarlanmış

En iyi parametreler: `C=1`, `gamma='auto'`, `kernel='rbf'`

### Test kümesi sonuçları (200 başvuru — eğitimde hiç kullanılmadı)

| Metrik | Değer |
|---|---|
| Accuracy | **%72.50** |
| Precision (riskli) | %52.94 |
| **Recall (riskli)** | **%75.00** |
| F1-Score | %62.07 |
| **ROC-AUC** | **%80.15** |

**Karmaşıklık matrisi:**

|  | Tahmin: İyi | Tahmin: Riskli |
|---|---|---|
| **Gerçek: İyi** | 100 (TN) | 40 (FP) |
| **Gerçek: Riskli** | 15 (FN) | 45 (TP) |

**Maliyet analizi:** Model `15×5 + 40×1 = 115` birim zarar üretiyor.
Hiç model kullanmayıp herkese kredi veren bir banka `60×5 = 300` birim kaybederdi.
→ **%62 daha az zarar.**

---

## 7. Sonuç Yorumu

### Metriği doğru seçmek, model seçmek kadar önemli

Hangi metriğe baktığınıza göre kazanan değişiyor:

- **En yüksek accuracy:** KNN (%71.5)
- **En yüksek recall:** Lojistik Regresyon (%70.0)
- **En yüksek ROC-AUC:** SVM (%74.4)

KNN bunu net gösteriyor: accuracy'si listenin üstünde ama **recall'ı yalnızca %20**.
Yani riskli müşterilerin beşte dördünü kaçırıyor ve yüksek accuracy'yi
"çoğunluğa iyi müşteri de" diyerek elde ediyor. Bir banka bu modeli kullansa batardı.

### Verinin söylediği iki karşı-sezgisel bulgu

**1. Vadesiz hesabı OLMAYAN müşteriler en güvenli grup.**
Risk oranı %11.7 — tüm gruplar içindeki en düşük değer. Bakiyesi eksiye düşenlerde
ise %49.3. "Hesabı yok" bir yoksunluk değil; bu bankada hiç eksi bakiye üretmemiş
olmak anlamına geliyor.

**2. Hiç kredi kullanmamış müşteriler en riskli grup.**
Risk oranı %62.5. Buna karşılık "kritik hesap / başka bankada kredisi olanlar"
grubunda oran %17.1. Kredi skorlamasında buna **"ince dosya" (thin file) problemi**
deniyor: ödeme geçmişi olmayan birinin güvenilirliğini kanıtlayacak hiçbir kaydı yoktur.

Sezgiyle hareket etseydik ikisini de ters kurgulardık. "Önce veriye bak, sonra
model kur" ilkesinin karşılığı tam olarak bu.

### İki önem ölçütü aynı şeyi söylemiyor

| Rastgele Orman (gini) ilk 3 | Permütasyon önemi ilk 3 |
|---|---|
| `aylik_taksit` (%7.7) | `hesap_durumu_vadesiz hesabı yok` (+3.94%) |
| `log_kredi_tutari` (%7.2) | `hesap_durumu_bakiye < 0 DM` (+1.98%) |
| `kredi_tutari` (%7.1) | `kredi_amaci_yeni araba` (+1.05%) |

İlk 5'te yalnızca 1 ortak değişken var. Fark tesadüf değil: Rastgele Orman'ın
dahilî önem skoru, çok sayıda farklı değer alan **sürekli** değişkenleri sistematik
olarak kayırır. Permütasyon önemi bu yanlılığı taşımıyor ve doğrudan
"bu sütunu bozarsam skorum ne kadar düşer?" sorusunu soruyor. Sonucu, veriden
gözle gördüğümüz bulgularla da örtüşüyor.

---

## 8. Etik Değerlendirme

Veri setindeki `medeni_durum_cinsiyet` sütunu müşterinin **cinsiyetini** içeriyor ve
lojistik regresyon katsayılarında riski en çok artıran değişken olarak çıkıyor.
Cinsiyete dayalı kredi kararı Türkiye dâhil pek çok ülkede yasa dışıdır.

Bu yüzden ölçtük:

| | ROC-AUC | Recall |
|---|---|---|
| Cinsiyet **dahil** | %80.15 | %75.00 |
| Cinsiyet **hariç** | %80.27 | %73.33 |

**Sonuç:** Cinsiyet bilgisini çıkarmanın modele maliyeti neredeyse sıfır.
Bu kadar küçük bir kazanç için ayrımcı bir değişkeni modelde tutmanın savunulabilir
bir gerekçesi yok. Projede sütun ana modelde bilerek bırakıldı ki etkisi ölçülebilsin;
**üretime alınacak bir modelde bulunmamalıdır.**

---

## 9. Modelin Sınırlılıkları

1. **Veri eski ve yereldir.** 1990'ların Almanya'sından, para birimi Alman Markı.
   Bugünün Türkiye'sindeki bir bankaya doğrudan uygulanamaz. Bu bir yöntem
   denemesidir, kullanıma hazır bir ürün değil.
2. **Veri küçük.** 1.000 satırın 200'ü test kümesinde. Bu boyutta birkaç örneklik
   farklar bile metrikleri gözle görülür oynatır — bu yüzden tek bir validation
   skoruna değil, 5 katlı çapraz doğrulamaya bakıldı.
3. **Sınıflar dengesiz.** `class_weight="balanced"` ile hafifletildi, ancak riskli
   sınıf hâlâ azınlıkta. Daha fazla riskli örnek toplamak, her türlü algoritma
   iyileştirmesinden daha etkili olurdu.
4. **Öznitelik seçimi tek modele dayanıyor.** Seçim Rastgele Orman'ın önem
   skorlarıyla yapıldı; farklı bir algoritma farklı bir alt küme seçebilirdi.
5. **Aykırı değerler silinmedi.** `kredi_tutari` sütunundaki 72 aykırı değer ölçüm
   hatası değil, gerçekten yüksek tutarlı başvurular. Bankanın en çok para
   kaybedebileceği başvurular tam olarak bunlar; silmek modeli en kritik vakalara
   kör bırakırdı. Bunun yerine log dönüşümü uygulandı.

---

## 10. Kaynaklar

- [UCI ML Repository — Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- [scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn — Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Türkiye Yapay Zekâ Akademisi — Makine Öğrenmesi ders deposu](https://github.com/turkiyeyapayzekaakademisi/makine-ogrenmesi)
