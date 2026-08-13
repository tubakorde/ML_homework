# -*- coding: utf-8 -*-
"""
================================================================================
KREDİ RİSKİ TAHMİNİ — Uçtan Uca Makine Öğrenmesi Projesi
Türkiye Yapay Zekâ Akademisi | Makine Öğrenmesi Final Ödevi
================================================================================

AMAÇ
    Bir bankaya kredi başvurusunda bulunan müşterinin "iyi" mi yoksa "riskli" mi
    olduğunu, başvuru anında bilinen bilgilerle önceden tahmin etmek.

    Problem türü : İKİLİ SINIFLANDIRMA (binary classification)
    Hedef        : risk  →  0 = iyi müşteri, 1 = riskli müşteri
    Veri seti    : UCI Statlog (German Credit Data) — 1.000 başvuru, 20 değişken

    Bu veri setinin özel bir yanı var: sahibi, hataların maliyetinin eşit
    olmadığını açıkça belirtmiş. Riskli bir müşteriye kredi vermek (FN),
    iyi bir müşteriyi geri çevirmekten (FP) 5 KAT daha pahalı. Proje boyunca
    metrik seçimlerimizi bu gerçeğe göre yapıyoruz — bu yüzden başarı ölçütümüz
    "accuracy" değil, riskli sınıfın yakalanma oranı (recall) ve toplam maliyet.

KULLANILAN KÜTÜPHANELER
    pandas, numpy        : veri okuma, dönüştürme, öznitelik üretme
    matplotlib           : grafiklerin çizilmesi
    scikit-learn         : ön işleme, 5 farklı model, çapraz doğrulama,
                           hiperparametre araması, değerlendirme metrikleri
    ucimlrepo (opsiyonel): veri seti yerelde yoksa UCI'dan indirmek için

ÇALIŞTIRMA ADIMLARI
    1) Sanal ortam oluşturun (önerilir):
           python -m venv .venv
           .venv\\Scripts\\activate          (Windows)
           source .venv/bin/activate         (macOS / Linux)
    2) Bağımlılıkları kurun:
           pip install -r requirements.txt
    3) Betiği çalıştırın:
           python kredi_riski_tahmini.py

    Veri dosyası (data/german_credit.csv) repoda mevcuttur. Silinmiş olsa bile
    betik veriyi UCI sunucusundan otomatik indirip aynı klasöre kaydeder.
    Üretilen tüm grafikler gorseller/ klasörüne yazılır.
    Toplam çalışma süresi ortalama bir dizüstü bilgisayarda ~1-2 dakikadır.

İZLENEN AKIŞ
    1.  Veriyi yükle ve problemi tanımla
    2.  Hedef değişkeni belirle
    3.  Veriyi incele (boyut, tipler, istatistikler)
    4.  Eksik değer analizi
    5.  Aykırı değer analizi (IQR)
    6.  Öznitelik mühendisliği — 5 yeni değişken
    7.  Kategorik değişkenleri sayısallaştır (one-hot encoding)
    8.  Train / validation / test ayrımı (stratify ile)
    9.  Sayısal değişkenleri ölçekle (yalnızca train üzerinde fit)
    10. Öznitelik seçimi (düşük varyans elemesi + korelasyon + önem sırası)
    11. 5 farklı model eğit
    12. Validation skorları ve çapraz doğrulama ile karşılaştır
    13. En iyi model için GridSearchCV
    14. Test kümesinde nihai değerlendirme
    15. Sonuçların yorumlanması
    16. Açıklanabilirlik (öznitelik önemi + permütasyon önemi + katsayılar)
================================================================================
"""

# ------------------------------------------------------------------ 1. KÜTÜPHANELER
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # grafikleri ekrana basmadan dosyaya yaz
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
VERI_YOLU = os.path.join(BASE, "data", "german_credit.csv")
GORSEL_KLASORU = os.path.join(BASE, "gorseller")
os.makedirs(GORSEL_KLASORU, exist_ok=True)

# grafik stili
YESIL, KIRMIZI, LACIVERT, GRI = "#2A9D8F", "#E76F51", "#264653", "#8D99AE"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def baslik(metin):
    """Konsol çıktısını bölümlere ayırarak okunabilir hale getirir."""
    print("\n" + "=" * 78)
    print(f"  {metin}")
    print("=" * 78)


def grafik_kaydet(fig, dosya_adi):
    yol = os.path.join(GORSEL_KLASORU, dosya_adi)
    fig.savefig(yol)
    plt.close(fig)
    print(f"   [grafik] gorseller/{dosya_adi}")


def sadelestir(ax):
    for kenar in ("top", "right"):
        ax.spines[kenar].set_visible(False)
    return ax


# ============================================================ 2. VERİYİ YÜKLE
baslik("2. VERİ SETİNİN YÜKLENMESİ VE PROBLEM TANIMI")

print("""
PROBLEM
   Bir banka, kredi başvurusunu değerlendirirken şu soruyu cevaplamak zorunda:
   "Bu müşteri krediyi geri öder mi?"

   Karar bugüne kadar büyük ölçüde uzman sezgisine dayanıyordu. Elimizdeki veri
   seti, 1.000 gerçek başvurunun hem başvuru anındaki bilgilerini hem de sonradan
   ortaya çıkan ödeme davranışını içeriyor. Yani modele "geçmişte kimler ödedi,
   kimler ödemedi" sorusunu öğretebiliriz.

VERİ SETİ
   UCI Machine Learning Repository — Statlog (German Credit Data)
   Prof. Dr. Hans Hofmann, Universität Hamburg tarafından derlenmiş.
   1.000 başvuru, 20 değişken (13 kategorik + 7 sayısal).
""")

if not os.path.exists(VERI_YOLU):
    print("   Yerel veri dosyası bulunamadı, UCI'dan indiriliyor...")
    from ucimlrepo import fetch_ucirepo
    _d = fetch_ucirepo(id=144)
    os.makedirs(os.path.dirname(VERI_YOLU), exist_ok=True)
    pd.concat([_d.data.features, _d.data.targets], axis=1).to_csv(VERI_YOLU, index=False)
    print("   İndirildi ve kaydedildi.")

df = pd.read_csv(VERI_YOLU)

# Sütun adları orijinalinde Attribute1..Attribute20 şeklinde; anlaşılır hale getiriyoruz.
SUTUN_ADLARI = {
    "Attribute1": "hesap_durumu",        "Attribute2": "vade_ay",
    "Attribute3": "kredi_gecmisi",       "Attribute4": "kredi_amaci",
    "Attribute5": "kredi_tutari",        "Attribute6": "birikim",
    "Attribute7": "calisma_suresi",      "Attribute8": "taksit_orani",
    "Attribute9": "medeni_durum_cinsiyet", "Attribute10": "kefil_durumu",
    "Attribute11": "ikamet_suresi",      "Attribute12": "mulk",
    "Attribute13": "yas",                "Attribute14": "diger_taksitler",
    "Attribute15": "konut_durumu",       "Attribute16": "bankadaki_kredi_sayisi",
    "Attribute17": "meslek",             "Attribute18": "bakmakla_yukumlu_kisi",
    "Attribute19": "telefon",            "Attribute20": "yabanci_isci",
    "class": "risk",
}
df = df.rename(columns=SUTUN_ADLARI)

# Kategorik kodlar (A11, A34 ...) tek başına okunmuyor; anlamlı karşılıklarını yazıyoruz.
KOD_SOZLUGU = {
    "hesap_durumu": {"A11": "bakiye < 0 DM", "A12": "0-200 DM", "A13": "200+ DM",
                     "A14": "vadesiz hesabı yok"},
    "kredi_gecmisi": {"A30": "hiç kredi almamış", "A31": "bu bankada hepsi ödendi",
                      "A32": "mevcut krediler düzenli", "A33": "geçmişte gecikme",
                      "A34": "kritik hesap / başka bankada kredi"},
    "kredi_amaci": {"A40": "yeni araba", "A41": "ikinci el araba", "A42": "mobilya",
                    "A43": "radyo/TV", "A44": "ev aleti", "A45": "onarım",
                    "A46": "eğitim", "A47": "tatil", "A48": "mesleki eğitim",
                    "A49": "iş kurma", "A410": "diğer"},
    "birikim": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                "A64": "1000+ DM", "A65": "birikimi yok / bilinmiyor"},
    "calisma_suresi": {"A71": "işsiz", "A72": "< 1 yıl", "A73": "1-4 yıl",
                       "A74": "4-7 yıl", "A75": "7+ yıl"},
    "medeni_durum_cinsiyet": {"A91": "erkek, boşanmış", "A92": "kadın, boşanmış/evli",
                              "A93": "erkek, bekar", "A94": "erkek, evli/dul",
                              "A95": "kadın, bekar"},
    "kefil_durumu": {"A101": "yok", "A102": "müşterek borçlu", "A103": "kefil"},
    "mulk": {"A121": "gayrimenkul", "A122": "hayat sigortası", "A123": "araba/diğer",
             "A124": "mülkü yok"},
    "diger_taksitler": {"A141": "banka", "A142": "mağaza", "A143": "yok"},
    "konut_durumu": {"A151": "kira", "A152": "ev sahibi", "A153": "ücretsiz konut"},
    "meslek": {"A171": "işsiz/vasıfsız", "A172": "vasıfsız", "A173": "vasıflı",
               "A174": "yönetici/nitelikli"},
    "telefon": {"A191": "yok", "A192": "var"},
    "yabanci_isci": {"A201": "evet", "A202": "hayır"},
}
for sutun, sozluk in KOD_SOZLUGU.items():
    df[sutun] = df[sutun].map(sozluk).fillna(df[sutun])

print(f"   Veri başarıyla yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")


# ======================================================= 3. HEDEF DEĞİŞKEN
baslik("3. HEDEF DEĞİŞKENİN BELİRLENMESİ")

# Orijinal kodlama: 1 = iyi müşteri, 2 = kötü müşteri.
# Modelleme kolaylığı için riskli sınıfı 1 (pozitif sınıf) yapıyoruz:
# böylece "recall" doğrudan "riskli müşterilerin kaçını yakaladık" anlamına geliyor.
df["risk"] = df["risk"].map({1: 0, 2: 1})

dagilim = df["risk"].value_counts().sort_index()
print(f"""
   Hedef değişken : 'risk'
   Problem türü   : İKİLİ SINIFLANDIRMA (binary classification)

   0 = iyi müşteri     : {dagilim[0]:>4} kişi  (%{dagilim[0]/len(df)*100:.1f})
   1 = riskli müşteri  : {dagilim[1]:>4} kişi  (%{dagilim[1]/len(df)*100:.1f})

   Sınıflar dengesiz. Hiçbir şey öğrenmeden herkese "iyi müşteri" diyen bir model
   %70 doğruluk alır. Bu yüzden accuracy tek başına anlamlı bir ölçüt değil —
   kurduğumuz her modelin bu %70'i geçmesi gerekiyor.
""")

fig, eksenler = plt.subplots(1, 2, figsize=(11, 4.2))
eksenler[0].bar(["İyi müşteri", "Riskli müşteri"], dagilim.values,
                color=[YESIL, KIRMIZI], width=0.55)
for i, v in enumerate(dagilim.values):
    eksenler[0].text(i, v + 12, f"{v}\n(%{v/len(df)*100:.0f})", ha="center",
                     fontweight="bold")
eksenler[0].set_title("Hedef değişken dağılımı")
eksenler[0].set_ylabel("Başvuru sayısı")
eksenler[0].set_ylim(0, 800)
sadelestir(eksenler[0])

oranlar = df.groupby("hesap_durumu")["risk"].mean().sort_values() * 100
eksenler[1].barh(oranlar.index, oranlar.values,
                 color=[KIRMIZI if v > 30 else YESIL for v in oranlar.values])
for i, v in enumerate(oranlar.values):
    eksenler[1].text(v + 1, i, f"%{v:.0f}", va="center", fontweight="bold")
eksenler[1].set_title("Vadesiz hesap durumuna göre risk oranı")
eksenler[1].set_xlabel("Riskli müşteri oranı (%)")
eksenler[1].set_xlim(0, 65)
sadelestir(eksenler[1])
grafik_kaydet(fig, "01_hedef_dagilimi.png")


# ======================================================= 4. VERİ İNCELEME
baslik("4. VERİ SETİNİN İNCELENMESİ")

print("\n--- İlk 5 satır (seçili sütunlar) ---")
print(df[["hesap_durumu", "vade_ay", "kredi_tutari", "yas",
          "kredi_gecmisi", "risk"]].head().to_string())

print(f"\n--- Boyut ---\n   Satır: {df.shape[0]}   Sütun: {df.shape[1]}")

print("\n--- Veri tipleri ---")
print(df.dtypes.value_counts().to_string())

SAYISAL_SUTUNLAR = ["vade_ay", "kredi_tutari", "taksit_orani", "ikamet_suresi",
                    "yas", "bankadaki_kredi_sayisi", "bakmakla_yukumlu_kisi"]
KATEGORIK_SUTUNLAR = [s for s in df.columns
                      if s not in SAYISAL_SUTUNLAR + ["risk"]]

print(f"\n   Sayısal değişken sayısı   : {len(SAYISAL_SUTUNLAR)}")
print(f"   Kategorik değişken sayısı : {len(KATEGORIK_SUTUNLAR)}")

print("\n--- Sayısal değişkenlerin temel istatistikleri ---")
print(df[SAYISAL_SUTUNLAR].describe().T.round(2).to_string())

# --- 4.1 Kategorik değişkenlerin riskle ilişkisi -----------------------------
# Modeli kurmadan önce veriye bakmanın karşılığı burada ortaya çıkıyor:
# bazı bulgular sezgiye tamamen ters.
print("\n--- 4.1 Kategorik değişkenlere göre riskli müşteri oranı ---")

kategori_riskleri = {}
for sutun in ["hesap_durumu", "kredi_gecmisi", "birikim", "calisma_suresi"]:
    ozet = (df.groupby(sutun)["risk"]
              .agg(risk_orani="mean", kisi="count")
              .sort_values("risk_orani"))
    ozet["risk_orani"] = (ozet["risk_orani"] * 100).round(1)
    kategori_riskleri[sutun] = ozet
    print(f"\n   [{sutun}]")
    for kategori, satir in ozet.iterrows():
        print(f"      {str(kategori):38s} %{satir['risk_orani']:>5.1f}   "
              f"({int(satir['kisi'])} kişi)")

# Sezgiye ters çıkan iki bulguyu programatik olarak yakalayalım
_hesap = kategori_riskleri["hesap_durumu"]
_gecmis = kategori_riskleri["kredi_gecmisi"]
print(f"""
   İKİ ŞAŞIRTICI BULGU

   1) "Vadesiz hesabı yok" grubunun risk oranı %{_hesap.loc['vadesiz hesabı yok', 'risk_orani']},
      yani TÜM GRUPLAR İÇİNDE EN DÜŞÜĞÜ. Bakiyesi eksiye düşenlerde ise
      %{_hesap.loc['bakiye < 0 DM', 'risk_orani']}. Sezgi "hesabı olmayan risklidir" der; veri tersini söylüyor.
      Muhtemel açıklama: bu kişiler bankacılığını başka kurumda yapıyor ve
      bu bankada hiç eksi bakiye üretmiyor. "Hesabı olmamak" bir yoksunluk
      değil, sadece görünmezlik.

   2) "Hiç kredi almamış" müşterilerin risk oranı %{_gecmis.loc['hiç kredi almamış', 'risk_orani']} —
      EN YÜKSEK grup. Buna karşılık "kritik hesap / başka bankada kredi"
      grubunda oran %{_gecmis.loc['kritik hesap / başka bankada kredi', 'risk_orani']}. Kredi skorlamasında buna "ince dosya"
      (thin file) problemi deniyor: ödeme geçmişi olmayan birinin ödeyip
      ödemeyeceğini kanıtlayacak hiçbir kaydı yoktur, bu da başlı başına risktir.

   Bu iki bulgu, modelin sonuçlarını yorumlarken aklımızda olacak.
""")

fig, eksenler = plt.subplots(1, 2, figsize=(14, 4.6))
for eksen, sutun, baslik_metni in zip(
        eksenler, ["hesap_durumu", "kredi_gecmisi"],
        ["Vadesiz hesap durumu", "Kredi geçmişi"]):
    ozet = kategori_riskleri[sutun]
    renkler = [KIRMIZI if v > 30 else YESIL for v in ozet["risk_orani"]]
    eksen.barh(ozet.index.astype(str), ozet["risk_orani"], color=renkler, height=0.62)
    for i, (v, n) in enumerate(zip(ozet["risk_orani"], ozet["kisi"])):
        eksen.text(v + 1, i, f"%{v:.0f}  (n={int(n)})", va="center", fontsize=8.5,
                   fontweight="bold")
    eksen.axvline(30, color=GRI, ls="--", lw=1.3)
    eksen.set_title(baslik_metni)
    eksen.set_xlabel("Riskli müşteri oranı (%)")
    eksen.set_xlim(0, 82)
    eksen.tick_params(axis="y", labelsize=8.5)
    sadelestir(eksen)
fig.suptitle("Kesikli çizgi: genel ortalama (%30). Sağında kalanlar riskli gruplar.",
             fontsize=10.5, color=GRI, style="italic", y=0.02)
grafik_kaydet(fig, "02_kategorik_risk.png")


# ======================================================= 5. EKSİK DEĞER
baslik("5. EKSİK DEĞER ANALİZİ")

eksikler = df.isnull().sum()
print(f"\n   Toplam eksik hücre sayısı: {eksikler.sum()}")

if eksikler.sum() == 0:
    print("""
   Veri setinde teknik anlamda hiç eksik değer yok. Ancak bu, "hiçbir bilgi
   eksik değil" demek DEĞİL. İki sütunda eksiklik, ayrı bir kategori olarak
   kodlanmış durumda:

       hesap_durumu = 'vadesiz hesabı yok'
       birikim      = 'birikimi yok / bilinmiyor'

   Bunlar dropna() ile yakalanamaz. Yine de silmiyoruz: "banka hesabı olmaması"
   gerçek ve güçlü bir sinyal. one-hot encoding sırasında kendi sütunlarını
   alacaklar ve model bu bilgiyi kullanabilecek.
""")
    for sutun in ["hesap_durumu", "birikim"]:
        gizli = df[sutun].str.contains("yok", na=False).sum()
        print(f"   {sutun:16s} → 'yok/bilinmiyor' olan kayıt: {gizli:>4} "
              f"(%{gizli/len(df)*100:.1f})")
else:
    for sutun in df.columns[eksikler > 0]:
        if sutun in SAYISAL_SUTUNLAR:
            df[sutun] = df[sutun].fillna(df[sutun].median())
        else:
            df[sutun] = df[sutun].fillna(df[sutun].mode()[0])
    print("   Sayısal sütunlar medyan, kategorik sütunlar mod ile dolduruldu.")


# ======================================================= 6. AYKIRI DEĞER
baslik("6. AYKIRI DEĞER ANALİZİ (IQR YÖNTEMİ)")

print()
aykiri_ozet = []
for sutun in SAYISAL_SUTUNLAR:
    q1, q3 = df[sutun].quantile(0.25), df[sutun].quantile(0.75)
    iqr = q3 - q1
    alt, ust = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    maske = (df[sutun] < alt) | (df[sutun] > ust)
    aykiri_ozet.append({"degisken": sutun, "aykiri_sayisi": int(maske.sum()),
                        "oran_%": round(maske.sum() / len(df) * 100, 1),
                        "alt_sinir": round(alt, 1), "ust_sinir": round(ust, 1)})
print(pd.DataFrame(aykiri_ozet).to_string(index=False))

print("""
   DİKKAT: 'bakmakla_yukumlu_kisi' sütunu yalnızca 1 ve 2 değerlerini alıyor.
   Bu durumda Q1 = Q3 = 1 olduğu için IQR sıfır çıkıyor ve 2 değerini alan
   her satır "aykırı" işaretleniyor. IQR yöntemi neredeyse ikili değişkenlerde
   anlamsızdır — buradaki %15,5'lik oran bir sorun değil, yöntemin kendi sınırı.

   KARAR: Aykırı değerleri SİLMİYORUZ.

   Gerekçe: 'kredi_tutari' sütunundaki 72 aykırı değer bir ölçüm hatası değil;
   bunlar gerçekten yüksek tutarlı kredi başvuruları. Bankanın en çok para
   kaybedebileceği başvurular tam olarak bunlar. Silmek, modeli en kritik
   vakalara kör bırakmak olur.

   Bunun yerine iki şey yapıyoruz:
     1) kredi_tutari'nın çarpık dağılımını log dönüşümüyle yumuşatıyoruz
        (bir sonraki bölümde 'log_kredi_tutari' özniteliği olarak).
        Log dönüşümü satır bazlı bir işlem olduğu için veri sızıntısı yaratmaz.
     2) Ağaç tabanlı modeller (Karar Ağacı, Rastgele Orman) aykırı değerlerden
        zaten etkilenmez; ölçeğe duyarlı modeller için StandardScaler kullanıyoruz.
""")

fig, eksenler = plt.subplots(1, 3, figsize=(14, 4))
for eksen, sutun in zip(eksenler, ["kredi_tutari", "vade_ay", "yas"]):
    kutu = eksen.boxplot([df[df["risk"] == 0][sutun], df[df["risk"] == 1][sutun]],
                         tick_labels=["İyi", "Riskli"], patch_artist=True,
                         widths=0.5, medianprops=dict(color="black", linewidth=1.6))
    for kutucuk, renk in zip(kutu["boxes"], [YESIL, KIRMIZI]):
        kutucuk.set_facecolor(renk)
        kutucuk.set_alpha(0.75)
    eksen.set_title(sutun)
    sadelestir(eksen)
fig.suptitle("Aykırı değerler ve risk sınıfına göre dağılım", fontsize=14,
             fontweight="bold", y=1.02)
grafik_kaydet(fig, "03_aykiri_degerler.png")


# =============================================== 7. ÖZNİTELİK MÜHENDİSLİĞİ
baslik("7. ÖZNİTELİK MÜHENDİSLİĞİ (5 YENİ DEĞİŞKEN)")

# 1) Aylık taksit yükü — 10.000 TL'lik 12 aylık kredi ile 48 aylık kredi
#    çok farklı risk taşır. Ham tutar tek başına bunu söylemiyor.
df["aylik_taksit"] = df["kredi_tutari"] / df["vade_ay"]

# 2) Kredinin yaşa oranı — genç bir müşterinin aynı tutarı üstlenmesi daha riskli.
df["kredi_yas_orani"] = df["kredi_tutari"] / df["yas"]

# 3) Log dönüşümlü kredi tutarı — sağa çarpık dağılımı simetrikleştirir,
#    ölçeğe duyarlı modellerin (Lojistik Regresyon, KNN, SVM) işini kolaylaştırır.
df["log_kredi_tutari"] = np.log1p(df["kredi_tutari"])

# 4) Toplam borç yükü — taksitin gelire oranı ile vadenin çarpımı:
#    "gelirinin ne kadarını, ne kadar süreyle bağlıyor?"
df["toplam_yuk"] = df["taksit_orani"] * df["vade_ay"]

# 5) Yaş grubu — yaşın etkisi doğrusal değil; genç ve çok yaşlı gruplar farklı davranır.
df["yas_grubu"] = pd.cut(df["yas"], bins=[0, 25, 35, 50, 100],
                         labels=["18-25", "26-35", "36-50", "50+"])

URETILEN = ["aylik_taksit", "kredi_yas_orani", "log_kredi_tutari", "toplam_yuk"]
SAYISAL_SUTUNLAR += URETILEN
KATEGORIK_SUTUNLAR += ["yas_grubu"]

print("\n   Üretilen öznitelikler ve riskle korelasyonları:")
uretilen_kor = {ozn: df[ozn].corr(df["risk"]) for ozn in URETILEN}
for ozn, kor in uretilen_kor.items():
    print(f"      {ozn:20s}  risk ile korelasyon = {kor:+.3f}")

print("\n   Karşılaştırma için ham değişkenler:")
ham_kor = {s: df[s].corr(df["risk"]) for s in ["kredi_tutari", "vade_ay", "yas"]}
for ozn, kor in ham_kor.items():
    print(f"      {ozn:20s}  risk ile korelasyon = {kor:+.3f}")

print("\n   Yaş grubuna göre riskli müşteri oranı:")
yas_riskleri = df.groupby("yas_grubu", observed=True)["risk"].mean() * 100
for grup, oran in yas_riskleri.items():
    print(f"      {str(grup):8s} → %{oran:.1f}")

# Yorumu sabit metin olarak yazmıyoruz; hesaplanan değerlerden üretiyoruz.
en_guclu_uretilen = max(uretilen_kor, key=lambda k: abs(uretilen_kor[k]))
en_zayif_uretilen = min(uretilen_kor, key=lambda k: abs(uretilen_kor[k]))
en_riskli_yas = yas_riskleri.idxmax()

print(f"""
   YORUM
   Ürettiğimiz beş öznitelik içinde riskle en güçlü doğrusal ilişkiyi
   '{en_guclu_uretilen}' kurdu ({uretilen_kor[en_guclu_uretilen]:+.3f}); ham 'kredi_tutari'
   ({ham_kor['kredi_tutari']:+.3f}) bunun gerisinde kaldı. Yani bankayı ilgilendiren şey
   sadece borcun büyüklüğü değil, gelirin ne kadarının ne kadar süreyle
   bağlandığı.

   Buna karşılık '{en_zayif_uretilen}' neredeyse sıfır korelasyon verdi
   ({uretilen_kor[en_zayif_uretilen]:+.3f}). Bu, özniteliğin işe yaramaz olduğu anlamına
   gelmez: korelasyon yalnızca DOĞRUSAL ilişkiyi ölçer. Ağaç tabanlı modeller
   doğrusal olmayan eşikler bulabilir, bu yüzden özniteliği elemeden önce
   11. bölümdeki önem sıralamasına bakacağız.

   Yaş grupları arasında en riskli grup {en_riskli_yas} (%{yas_riskleri.max():.1f}),
   en güvenli grup {yas_riskleri.idxmin()} (%{yas_riskleri.min():.1f}). Yaşın etkisi
   doğrusal değil — bu yüzden ham 'yas' yanına gruplanmış hâlini de ekledik.
""")


# ==================================================== 8. KATEGORİK ENCODING
baslik("8. KATEGORİK DEĞİŞKENLERİN SAYISALLAŞTIRILMASI")

X = df.drop(columns=["risk"])
y = df["risk"]

print(f"\n   Encoding öncesi sütun sayısı : {X.shape[1]}")
X = pd.get_dummies(X, columns=KATEGORIK_SUTUNLAR, drop_first=True, dtype=int)
print(f"   Encoding sonrası sütun sayısı: {X.shape[1]}")
print("""
   Yöntem: one-hot encoding (pd.get_dummies).

   Neden label encoding değil? Label encoding kategorilere 0,1,2,3 gibi sayılar
   atar ve model bunlar arasında sıralama/uzaklık olduğunu sanır. 'kredi_amaci'
   sütununda "araba" = 1, "eğitim" = 2 olsaydı, model "eğitim, arabanın iki katı"
   gibi anlamsız bir ilişki kurardı. one-hot encoding bu tuzağı ortadan kaldırır.

   drop_first=True ile her değişkenden bir kategori referans olarak düşürülüyor;
   bu, sütunlar arası tam doğrusal bağımlılığı (dummy variable trap) önler.
""")


# ============================================ 9. TRAIN / VALIDATION / TEST
baslik("9. VERİNİN TRAIN / VALIDATION / TEST OLARAK AYRILMASI")

# Önce test kümesini ayırıyoruz (%20), sonra kalanı train/validation olarak bölüyoruz.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

print(f"""
   Train      : {X_train.shape[0]:>4} satır  (%{X_train.shape[0]/len(X)*100:.0f})  → modeller burada öğreniyor
   Validation : {X_val.shape[0]:>4} satır  (%{X_val.shape[0]/len(X)*100:.0f})  → model seçimi burada yapılıyor
   Test       : {X_test.shape[0]:>4} satır  (%{X_test.shape[0]/len(X)*100:.0f})  → yalnızca EN SONDA bir kez kullanılıyor

   stratify=y kullanıldı: üç kümenin de sınıf oranı orijinal veriyle aynı kalıyor.
""")
for ad, hedef in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
    print(f"   {ad:11s} riskli oranı: %{hedef.mean()*100:.1f}")


# ======================================================= 10. ÖLÇEKLEME
baslik("10. SAYISAL DEĞİŞKENLERİN ÖLÇEKLENMESİ")

olcekleyici = StandardScaler()
X_train_olcekli = X_train.copy()
X_val_olcekli = X_val.copy()
X_test_olcekli = X_test.copy()

# ÖNEMLİ: fit YALNIZCA train üzerinde yapılır. Validation ve test yalnızca
# transform edilir. Aksi hâlde test verisinin ortalaması modele sızar (data leakage).
X_train_olcekli[SAYISAL_SUTUNLAR] = olcekleyici.fit_transform(X_train[SAYISAL_SUTUNLAR])
X_val_olcekli[SAYISAL_SUTUNLAR] = olcekleyici.transform(X_val[SAYISAL_SUTUNLAR])
X_test_olcekli[SAYISAL_SUTUNLAR] = olcekleyici.transform(X_test[SAYISAL_SUTUNLAR])

print(f"""
   StandardScaler uygulandı — her değişken ortalama 0, standart sapma 1 olacak
   şekilde dönüştürüldü. Ölçekleyici SADECE train verisi üzerinde fit edildi.

   Ölçekleme neden gerekli? KNN ve SVM uzaklık hesabı yapar. 'kredi_tutari'
   binlerle, 'taksit_orani' 1-4 arasında değer alıyor. Ölçeklemeseydik uzaklık
   neredeyse tamamen kredi tutarına göre belirlenirdi.

   Ölçeklenmiş train ortalaması : {X_train_olcekli[SAYISAL_SUTUNLAR].mean().mean():.3f}  (~0 bekleniyor)
   Ölçeklenmiş train std. sapma : {X_train_olcekli[SAYISAL_SUTUNLAR].std().mean():.3f}  (~1 bekleniyor)
""")


# =================================================== 11. ÖZNİTELİK SEÇİMİ
baslik("11. ÖZNİTELİK SEÇİMİ (ÜÇ AŞAMALI)")

# --- Aşama 1: düşük varyanslı (neredeyse sabit) sütunları ele
varyans_esigi = VarianceThreshold(threshold=0.01)
varyans_esigi.fit(X_train)
dusuk_varyansli = X_train.columns[~varyans_esigi.get_support()].tolist()
print(f"\n   [1] Düşük varyans elemesi → {len(dusuk_varyansli)} sütun elendi")
for s in dusuk_varyansli:
    print(f"       - {s}")

# --- Aşama 2: hedefle korelasyon (yalnızca train üzerinden)
kalan = [s for s in X_train.columns if s not in dusuk_varyansli]
korelasyonlar = (X_train[kalan].corrwith(y_train)
                 .abs().sort_values(ascending=False))
print(f"\n   [2] Hedefle mutlak korelasyonu en yüksek 8 değişken:")
for ad, deger in korelasyonlar.head(8).items():
    print(f"       {ad:42s} {deger:.3f}")

# --- Aşama 3: Rastgele Orman öznitelik önemi ile nihai seçim
onem_modeli = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
onem_modeli.fit(X_train[kalan], y_train)
onemler = pd.Series(onem_modeli.feature_importances_, index=kalan).sort_values(ascending=False)

# Kümülatif önemin %95'ini açıklayan öznitelikleri tutuyoruz
kumulatif = onemler.cumsum() / onemler.sum()
SECILEN = kumulatif[kumulatif <= 0.95].index.tolist()
if len(SECILEN) < 10:
    SECILEN = onemler.head(25).index.tolist()

print(f"\n   [3] Rastgele Orman önem sırası → kümülatif %95'i açıklayan "
      f"{len(SECILEN)} öznitelik seçildi")
print(f"       (toplam {X_train.shape[1]} sütundan {len(SECILEN)} sütuna indi)")
print("\n   En önemli 10 öznitelik:")
for ad, deger in onemler.head(10).items():
    isaret = "✓" if ad in SECILEN else " "
    print(f"     {isaret} {ad:42s} {deger*100:.2f}%")

fig, eksen = plt.subplots(figsize=(9, 6.5))
ilk15 = onemler.head(15)[::-1]
eksen.barh(ilk15.index, ilk15.values * 100,
           color=plt.cm.YlOrRd(np.linspace(0.35, 0.85, len(ilk15))))
for i, v in enumerate(ilk15.values * 100):
    eksen.text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
eksen.set_title("Öznitelik önemi — en etkili 15 değişken (Rastgele Orman)")
eksen.set_xlabel("Önem (%)")
sadelestir(eksen)
grafik_kaydet(fig, "04_oznitelik_onemi.png")

# Seçilen özniteliklerle çalışacak kümeler
X_train_son = X_train_olcekli[SECILEN]
X_val_son = X_val_olcekli[SECILEN]
X_test_son = X_test_olcekli[SECILEN]


# ======================================================= 12. MODEL EĞİTİMİ
baslik("12. BEŞ FARKLI MODELİN EĞİTİLMESİ")

# class_weight="balanced": azınlıkta kalan riskli sınıfa daha fazla ağırlık verir.
# Bu, "herkese iyi müşteri de, %70 tuttur" tembelliğini kırar.
MODELLER = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, class_weight="balanced",
                                             random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Karar Ağacı": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                          class_weight="balanced", random_state=42),
    "Rastgele Orman": RandomForestClassifier(n_estimators=300, max_depth=10,
                                             min_samples_leaf=5, class_weight="balanced",
                                             random_state=42, n_jobs=-1),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True,
                     class_weight="balanced", random_state=42),
}

print()
for ad, model in MODELLER.items():
    model.fit(X_train_son, y_train)
    print(f"   [egitildi] {ad}")


# ====================================== 13. VALIDATION + ÇAPRAZ DOĞRULAMA
baslik("13. MODEL KARŞILAŞTIRMASI (VALIDATION + ÇAPRAZ DOĞRULAMA)")

print("""
   METRİK SEÇİMİ
   Sınıflar dengesiz (%70 / %30) ve hataların maliyeti eşit değil. Bu yüzden
   modelleri accuracy ile değil, ağırlıklı olarak şu iki ölçütle karşılaştırıyoruz:

     • Recall (riskli sınıf) : gerçekten riskli müşterilerin kaçını yakaladık?
                               Bankanın para kaybettiği hata tam olarak burada.
     • ROC-AUC              : modelin sıralama gücü; eşik değerinden bağımsız.
""")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sonuclar = []

for ad, model in MODELLER.items():
    tahmin = model.predict(X_val_son)
    olasilik = model.predict_proba(X_val_son)[:, 1]

    # 5 katlı stratified çapraz doğrulama (yalnızca train verisi üzerinde)
    cv_skorlari = cross_val_score(model, X_train_son, y_train,
                                  cv=kfold, scoring="roc_auc", n_jobs=-1)

    sonuclar.append({
        "Model": ad,
        "Accuracy": accuracy_score(y_val, tahmin) * 100,
        "Precision": precision_score(y_val, tahmin, zero_division=0) * 100,
        "Recall": recall_score(y_val, tahmin) * 100,
        "F1": f1_score(y_val, tahmin) * 100,
        "ROC-AUC": roc_auc_score(y_val, olasilik) * 100,
        "CV AUC (ort)": cv_skorlari.mean() * 100,
        "CV AUC (std)": cv_skorlari.std() * 100,
    })

sonuc_df = pd.DataFrame(sonuclar).set_index("Model").round(2)
print("\n--- Validation kümesi sonuçları (%) ---")
print(sonuc_df.to_string())

EN_IYI_AD = sonuc_df["ROC-AUC"].idxmax()
print(f"""
   Çapraz doğrulama, tek bir validation bölünmesine güvenmemizi engelliyor.
   Standart sapma sütunu küçükse model kararlı demektir.

   → En yüksek ROC-AUC: {EN_IYI_AD}
     Hiperparametre ayarlamasını bu model üzerinde yapacağız.
""")

fig, eksenler = plt.subplots(1, 2, figsize=(14, 5))
eksen = eksenler[0]
x_konum = np.arange(len(sonuc_df))
genislik = 0.26
for kayma, kolon, renk in [(-genislik, "Accuracy", LACIVERT),
                           (0, "Recall", KIRMIZI),
                           (genislik, "ROC-AUC", YESIL)]:
    cubuklar = eksen.bar(x_konum + kayma, sonuc_df[kolon], genislik,
                         label=kolon, color=renk)
    for c in cubuklar:
        eksen.text(c.get_x() + c.get_width()/2, c.get_height() + 1,
                   f"{c.get_height():.0f}", ha="center", fontsize=8.5)
eksen.axhline(70, color=GRI, ls="--", lw=1.3)
eksen.text(len(sonuc_df) - 0.4, 71.5, "%70 taban çizgi", color=GRI, fontsize=9,
           ha="right", style="italic")
eksen.set_xticks(x_konum)
eksen.set_xticklabels(sonuc_df.index, rotation=18, ha="right", fontsize=9)
eksen.set_ylabel("Skor (%)")
eksen.set_ylim(0, 105)
eksen.set_title("Validation performansı")
eksen.legend(ncol=3, fontsize=9)
sadelestir(eksen)

eksen = eksenler[1]
eksen.bar(x_konum, sonuc_df["CV AUC (ort)"], 0.5, yerr=sonuc_df["CV AUC (std)"],
          capsize=5, color=YESIL, alpha=0.85,
          error_kw=dict(ecolor=LACIVERT, lw=1.4))
for i, v in enumerate(sonuc_df["CV AUC (ort)"]):
    eksen.text(i, v + 2.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
eksen.set_xticks(x_konum)
eksen.set_xticklabels(sonuc_df.index, rotation=18, ha="right", fontsize=9)
eksen.set_ylabel("ROC-AUC (%)")
eksen.set_ylim(0, 100)
eksen.set_title("5 katlı çapraz doğrulama (ortalama ± std)")
sadelestir(eksen)
grafik_kaydet(fig, "05_model_karsilastirma.png")


# ============================================ 14. HİPERPARAMETRE AYARLAMA
baslik("14. HİPERPARAMETRE AYARLAMA (GRID SEARCH)")

ARAMA_UZAYLARI = {
    "Rastgele Orman": (
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [6, 10, None],
         "min_samples_leaf": [1, 5, 10], "max_features": ["sqrt", "log2"]}),
    "Lojistik Regresyon": (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]}),
    "SVM (RBF)": (
        SVC(probability=True, class_weight="balanced", random_state=42),
        {"C": [0.1, 0.5, 1, 3, 10, 30],
         "gamma": ["scale", "auto", 0.005, 0.01, 0.05, 0.1],
         "kernel": ["rbf"]}),
    "Karar Ağacı": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"max_depth": [3, 5, 7, 10], "min_samples_leaf": [5, 10, 20],
         "criterion": ["gini", "entropy"]}),
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [5, 11, 15, 21], "weights": ["uniform", "distance"],
         "metric": ["euclidean", "manhattan"]}),
}

temel_model, parametre_izgarasi = ARAMA_UZAYLARI[EN_IYI_AD]
print(f"\n   Aranan model : {EN_IYI_AD}")
print(f"   Parametreler : {parametre_izgarasi}")

izgara_arama = GridSearchCV(estimator=temel_model, param_grid=parametre_izgarasi,
                            cv=kfold, scoring="roc_auc", n_jobs=-1, verbose=0)
izgara_arama.fit(X_train_son, y_train)

EN_IYI_MODEL = izgara_arama.best_estimator_

print(f"""
   Denenen kombinasyon sayısı : {len(izgara_arama.cv_results_['params'])}
   En iyi parametreler        : {izgara_arama.best_params_}
   En iyi CV ROC-AUC          : %{izgara_arama.best_score_*100:.2f}
""")

ayar_oncesi = sonuc_df.loc[EN_IYI_AD, "ROC-AUC"]
val_olasilik = EN_IYI_MODEL.predict_proba(X_val_son)[:, 1]
ayar_sonrasi = roc_auc_score(y_val, val_olasilik) * 100
print(f"   Validation ROC-AUC — ayar öncesi: %{ayar_oncesi:.2f}"
      f"  →  ayar sonrası: %{ayar_sonrasi:.2f}")

if abs(ayar_sonrasi - ayar_oncesi) < 0.5:
    print("""
   NOT: Arama sonucunda skor neredeyse hiç değişmedi. Bu bir başarısızlık
   değil, anlamlı bir bulgu: scikit-learn'ün varsayılan parametreleri bu veri
   seti için zaten iyi seçilmiş durumda. Hiperparametre ayarlaması her zaman
   büyük sıçrama getirmez; getirmediğinde bunu dürüstçe raporlamak, uydurma
   bir iyileşme aramaktan daha değerlidir.
""")


# ================================================= 15. TEST DEĞERLENDİRMESİ
baslik("15. NİHAİ MODELİN TEST KÜMESİNDE DEĞERLENDİRİLMESİ")

test_tahmin = EN_IYI_MODEL.predict(X_test_son)
test_olasilik = EN_IYI_MODEL.predict_proba(X_test_son)[:, 1]

test_accuracy = accuracy_score(y_test, test_tahmin) * 100
test_precision = precision_score(y_test, test_tahmin, zero_division=0) * 100
test_recall = recall_score(y_test, test_tahmin) * 100
test_f1 = f1_score(y_test, test_tahmin) * 100
test_auc = roc_auc_score(y_test, test_olasilik) * 100

print(f"""
   Model : {EN_IYI_AD} (GridSearchCV ile ayarlanmış)
   Test kümesi: {len(y_test)} başvuru — bu veri eğitim boyunca hiç kullanılmadı.

   Accuracy  : %{test_accuracy:.2f}
   Precision : %{test_precision:.2f}   (riskli dediklerimizin kaçı gerçekten riskli)
   Recall    : %{test_recall:.2f}   (gerçek risklilerin kaçını yakaladık)
   F1-Score  : %{test_f1:.2f}
   ROC-AUC   : %{test_auc:.2f}
""")

print("--- Sınıf bazlı rapor ---")
print(classification_report(y_test, test_tahmin,
                            target_names=["İyi müşteri", "Riskli müşteri"], digits=3))

kmatris = confusion_matrix(y_test, test_tahmin)
tn, fp, fn, tp = kmatris.ravel()
print("--- Karmaşıklık matrisi ---")
print(f"""
                        TAHMİN
                  İyi          Riskli
   GERÇEK  İyi    {tn:>4} (TN)    {fp:>4} (FP)
        Riskli    {fn:>4} (FN)    {tp:>4} (TP)
""")

# Veri setinin resmî maliyet matrisi: riskliye kredi vermek 5, iyiyi reddetmek 1 birim
maliyet = fn * 5 + fp * 1
naif_maliyet = int(y_test.sum()) * 5      # herkese "iyi müşteri" diyen model
print(f"""   MALİYET ANALİZİ (veri setinin resmî maliyet matrisiyle)
   Riskliye kredi vermek (FN) = 5 birim  |  İyiyi reddetmek (FP) = 1 birim

   Modelimizin toplam maliyeti      : {fn} × 5 + {fp} × 1 = {maliyet} birim
   Herkese "iyi müşteri" diyen model: {int(y_test.sum())} × 5 = {naif_maliyet} birim
   Kazanç                           : %{(1 - maliyet/naif_maliyet)*100:.1f} daha az zarar
""")

fig, eksenler = plt.subplots(1, 2, figsize=(13, 5.2))
eksen = eksenler[0]
goruntu = eksen.imshow(kmatris, cmap="Blues")
etiketler = ["İyi", "Riskli"]
eksen.set_xticks([0, 1], etiketler)
eksen.set_yticks([0, 1], etiketler)
kisaltmalar = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        eksen.text(j, i, f"{kmatris[i, j]}\n({kisaltmalar[i][j]})", ha="center",
                   va="center", fontsize=15, fontweight="bold",
                   color="white" if kmatris[i, j] > kmatris.max()/2 else LACIVERT)
eksen.set_xlabel("Modelin tahmini")
eksen.set_ylabel("Gerçek durum")
eksen.set_title(f"Karmaşıklık matrisi — {EN_IYI_AD}")
eksen.grid(False)

eksen = eksenler[1]
fpr, tpr, _ = roc_curve(y_test, test_olasilik)
eksen.plot(fpr, tpr, color=KIRMIZI, lw=2.4, label=f"ROC-AUC = {test_auc:.1f}%")
eksen.plot([0, 1], [0, 1], color=GRI, ls="--", lw=1.3, label="Rastgele tahmin (%50)")
eksen.fill_between(fpr, tpr, alpha=0.12, color=KIRMIZI)
eksen.set_xlabel("Yanlış alarm oranı (FPR)")
eksen.set_ylabel("Yakalama oranı (TPR)")
eksen.set_title("ROC eğrisi — test kümesi")
eksen.legend(loc="lower right")
sadelestir(eksen)
grafik_kaydet(fig, "06_test_sonuclari.png")


# ============================== 15.1 ETİK KONTROL: CİNSİYET DEĞİŞKENİ
baslik("15.1 ETİK KONTROL — CİNSİYET DEĞİŞKENİ OLMADAN MODEL")

print("""
   Veri setindeki 'medeni_durum_cinsiyet' sütunu müşterinin cinsiyetini
   içeriyor. Cinsiyete dayalı kredi kararı Türkiye dâhil pek çok ülkede
   yasa dışıdır. Bu yüzden şu soruyu ölçmek zorundayız:

       Bu sütunu tamamen çıkarırsak model ne kadar kötüleşiyor?

   Eğer kayıp küçükse, sütunu kullanmanın hiçbir savunulabilir gerekçesi kalmaz.
""")

cinsiyet_sutunlari = [s for s in SECILEN if s.startswith("medeni_durum_cinsiyet")]
SECILEN_ETIK = [s for s in SECILEN if s not in cinsiyet_sutunlari]

print(f"   Çıkarılan sütunlar ({len(cinsiyet_sutunlari)} adet):")
for s in cinsiyet_sutunlari:
    print(f"      - {s}")

# clone(): aynı model tipini ve aynı hiperparametreleri kopyalar, ama
# öğrenilmiş ağırlıkları taşımaz. Böylece adil bir karşılaştırma yapıyoruz.
from sklearn.base import clone
etik_model = clone(izgara_arama.best_estimator_)
etik_model.fit(X_train_olcekli[SECILEN_ETIK], y_train)

etik_tahmin = etik_model.predict(X_test_olcekli[SECILEN_ETIK])
etik_olasilik = etik_model.predict_proba(X_test_olcekli[SECILEN_ETIK])[:, 1]
etik_auc = roc_auc_score(y_test, etik_olasilik) * 100
etik_recall = recall_score(y_test, etik_tahmin) * 100
etik_fark = etik_auc - test_auc

print(f"""
   Cinsiyet DAHİL   → ROC-AUC %{test_auc:.2f}   Recall %{test_recall:.2f}
   Cinsiyet HARİÇ   → ROC-AUC %{etik_auc:.2f}   Recall %{etik_recall:.2f}
   Fark             → {etik_fark:+.2f} puan ROC-AUC

   SONUÇ: Cinsiyet bilgisini çıkarmanın modele maliyeti {abs(etik_fark):.2f} puan.
   {'Bu kayıp ihmal edilebilir düzeyde.' if abs(etik_fark) < 2 else 'Kayıp gözle görülür olsa da'}
   Ayrımcı bir değişkeni modelde tutmanın bedeli, elde edilen skordan çok daha
   ağırdır. Gerçek bir kredi sisteminde bu sütun veri setinden tamamen
   çıkarılmalı; ayrıca modelin cinsiyet gruplarına göre onay oranları düzenli
   olarak denetlenmelidir (fairness audit).

   Bu projede sütunu ana modelde bilerek bıraktık ki etkisini ölçebilelim;
   üretime alınacak bir modelde bırakılmamalıdır.
""")


# ======================================================= 16. YORUMLAMA
baslik("16. SONUÇLARIN YORUMLANMASI")

# Yorumları elle yazmak yerine gerçek sonuçlardan üretiyoruz; böylece veri
# ya da parametreler değiştiğinde metin de kendiliğinden güncelleniyor.
en_iyi_accuracy = sonuc_df["Accuracy"].idxmax()
en_iyi_recall = sonuc_df["Recall"].idxmax()
en_iyi_auc = sonuc_df["ROC-AUC"].idxmax()
en_kararli = sonuc_df["CV AUC (std)"].idxmin()
ilk_bes_onem = onemler.head(5)

print(f"""
HANGİ MODEL DAHA İYİ OLDU?
   Validation ROC-AUC sıralamasında {EN_IYI_AD} önde bitirdi; hiperparametre
   ayarından sonra test kümesinde %{test_auc:.2f} ROC-AUC ve %{test_recall:.2f} recall verdi.

   Ama asıl öğretici olan şu: HANGİ METRİĞE BAKTIĞINIZA GÖRE KAZANAN DEĞİŞİYOR.

     En yüksek Accuracy : {en_iyi_accuracy:22s} (%{sonuc_df.loc[en_iyi_accuracy, 'Accuracy']:.1f})
     En yüksek Recall   : {en_iyi_recall:22s} (%{sonuc_df.loc[en_iyi_recall, 'Recall']:.1f})
     En yüksek ROC-AUC  : {en_iyi_auc:22s} (%{sonuc_df.loc[en_iyi_auc, 'ROC-AUC']:.1f})
     En kararlı (en düşük CV std) : {en_kararli} (±{sonuc_df.loc[en_kararli, 'CV AUC (std)']:.1f})

   KNN'in durumu bunu çok net gösteriyor: accuracy'si %{sonuc_df.loc['KNN', 'Accuracy']:.1f} ile
   listenin üstlerinde, ama recall'ı yalnızca %{sonuc_df.loc['KNN', 'Recall']:.1f}. Yani riskli
   müşterilerin beşte dördünü kaçırıyor ve yüksek accuracy'yi "çoğunluğa iyi
   müşteri de" diyerek elde ediyor. Bir banka bu modeli kullansa batardı.

   Bu yüzden model seçimini ROC-AUC ve recall üzerinden yaptık. Metriği doğru
   seçmek, model seçmek kadar belirleyici.

HANGİ DEĞİŞKENLER ÖNEMLİ GÖRÜNÜYOR?
   Rastgele Orman'ın önem sıralamasına göre ilk beş:""")

for sira, (ad, deger) in enumerate(ilk_bes_onem.items(), 1):
    print(f"     {sira}. {ad:45s} %{deger*100:.2f}")

print(f"""
   Listenin tepesindeki isimlerin çoğu SAYISAL değişkenler (ve bizim ürettiğimiz
   öznitelikler). Bu kısmen bir yanılsama: Rastgele Orman'ın 'gini importance'
   ölçütü, çok sayıda farklı değer alan sürekli değişkenleri kayırır. İkili
   (0/1) kodlanmış kategorik sütunlar bu yarışta doğal olarak dezavantajlı.
   17. bölümdeki permütasyon önemi bu yanlılığı taşımıyor — iki listeyi
   karşılaştırmak öğretici olacak.

   VERİNİN SÖYLEDİĞİ İKİ TERS BULGU (4.1'de hesaplanmıştı)

   1. Vadesiz hesabı OLMAYAN müşteriler en güvenli grup
      (%{kategori_riskleri['hesap_durumu'].loc['vadesiz hesabı yok', 'risk_orani']} risk), bakiyesi eksiye düşenler ise en riskli
      (%{kategori_riskleri['hesap_durumu'].loc['bakiye < 0 DM', 'risk_orani']}). "Hesabı yok" bir yoksunluk değil, bu bankada hiç
      eksi bakiye üretmemiş olmak anlamına geliyor.

   2. Hiç kredi kullanmamış müşteriler en riskli grup
      (%{kategori_riskleri['kredi_gecmisi'].loc['hiç kredi almamış', 'risk_orani']}), "kritik hesap / başka bankada kredisi olanlar" ise
      en güvenli (%{kategori_riskleri['kredi_gecmisi'].loc['kritik hesap / başka bankada kredi', 'risk_orani']}). Kredi skorlamasındaki "ince dosya" problemi:
      ödeme geçmişi olmayan birinin güvenilirliğini kanıtlayacak veri yoktur.

   Bu iki bulgu, "önce veriye bak, sonra model kur" ilkesinin karşılığı.
   Sezgiyle hareket etseydik ikisini de ters kurgulardık.

MALİYET AÇISINDAN NE KAZANDIK?
   Modelin toplam maliyeti {maliyet} birim; hiç model kullanmayıp herkese kredi
   veren bir bankanınki {naif_maliyet} birim. Yani %{(1 - maliyet/naif_maliyet)*100:.0f} daha az zarar.
   {tp} riskli müşteriyi önceden yakaladık; {fn} tanesini kaçırdık.

MODELİN SINIRLILIKLARI
   1. VERİ ESKİ VE YERELDİR. Veri seti 1990'ların Almanya'sından; para birimi
      Alman Markı. Bugünün Türkiye'sindeki bir bankaya doğrudan uygulanamaz.
      Bu proje bir yöntem denemesidir, kullanıma hazır bir ürün değil.

   2. VERİ KÜÇÜK. 1.000 satırın 200'ü test kümesinde. Bu boyutta birkaç
      örneklik farklar bile metrikleri gözle görülür oynatır; bu yüzden tek bir
      validation skoruna değil, 5 katlı çapraz doğrulamaya baktık.

   3. ETİK SORUN — CİNSİYET DEĞİŞKENİ. 15.1'de ölçtüğümüz gibi, cinsiyet
      sütununu çıkarmanın modele maliyeti yalnızca {abs(etik_fark):.2f} puan ROC-AUC.
      Bu kadar küçük bir kazanç için ayrımcı bir değişkeni modelde tutmanın
      savunulabilir bir gerekçesi yok. Üretime alınacak bir modelde bu sütun
      bulunmamalı, ayrıca gruplar arası onay oranları denetlenmelidir.

   4. DENGESİZ SINIFLAR. class_weight="balanced" ile hafifletildi, ancak riskli
      sınıf hâlâ azınlıkta. Daha fazla riskli örnek toplamak, her türlü
      algoritma iyileştirmesinden daha etkili olurdu.

   5. ÖZNİTELİK SEÇİMİ TEK MODELE DAYANIYOR. Seçimi Rastgele Orman'ın önem
      skorlarıyla yaptık. Farklı bir algoritma farklı bir alt küme seçebilirdi.
""")


# =============================================== 17. AÇIKLANABİLİRLİK (BONUS)
baslik("17. AÇIKLANABİLİRLİK — MODEL KARARINI NASIL VERİYOR? (BONUS)")

# --- Permütasyon önemi: bir sütunu rastgele karıştırınca skor ne kadar düşüyor?
print("\n   [A] PERMÜTASYON ÖNEMİ (test kümesi üzerinde hesaplandı)")
print("       Bir sütunun değerleri rastgele karıştırıldığında modelin ROC-AUC")
print("       skoru ne kadar düşüyor? Ne kadar çok düşerse, o sütun o kadar kritik.\n")

perm = permutation_importance(EN_IYI_MODEL, X_test_son, y_test,
                              n_repeats=20, random_state=42,
                              scoring="roc_auc", n_jobs=-1)
perm_onem = pd.Series(perm.importances_mean, index=SECILEN).sort_values(ascending=False)

for ad, deger in perm_onem.head(10).items():
    std = perm.importances_std[list(SECILEN).index(ad)]
    print(f"       {ad:42s} {deger*100:+6.2f}%  (±{std*100:.2f})")

# İki önem ölçütünü karşılaştıralım — 16. bölümde bahsettiğimiz yanlılık burada görünür.
rf_ilk5 = set(onemler.head(5).index)
perm_ilk5 = set(perm_onem.head(5).index)
ortak = rf_ilk5 & perm_ilk5

print(f"""
       İKİ ÖNEM ÖLÇÜTÜNÜN KARŞILAŞTIRILMASI
       Rastgele Orman'ın ilk 5'i : {', '.join(list(onemler.head(5).index)[:5])}
       Permütasyonun ilk 5'i     : {', '.join(list(perm_onem.head(5).index)[:5])}
       Ortak olan               : {len(ortak)} / 5

       Fark tesadüf değil. Rastgele Orman'ın dahilî önem skoru, çok sayıda
       farklı değer alan SÜREKLİ değişkenleri sistematik olarak kayırır
       (bölünme noktası arayacak daha çok yer bulur). Permütasyon önemi ise
       doğrudan "bu sütunu bozarsam skorum ne kadar düşer?" sorusunu sorar
       ve bu yanlılığı taşımaz.

       Sonuç: permütasyon listesinde '{perm_onem.index[0]}' tepeye çıkıyor —
       bu, 4.1'de veriden gözümüzle gördüğümüz bulguyla da örtüşüyor.
       Model açıklarken hangi ölçütü kullandığınızı belirtmek şart.""")

# --- Lojistik regresyon katsayıları: yönü de gösteriyor
print("\n   [B] LOJİSTİK REGRESYON KATSAYILARI (yön bilgisi için)")
print("       Pozitif katsayı = riski ARTIRIYOR, negatif = riski AZALTIYOR.\n")

lojistik = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
lojistik.fit(X_train_son, y_train)
katsayilar = pd.Series(lojistik.coef_[0], index=SECILEN).sort_values()

print("       Riski en çok ARTIRAN 5 değişken:")
for ad, deger in katsayilar.tail(5)[::-1].items():
    print(f"         {ad:42s} {deger:+.3f}")
print("\n       Riski en çok AZALTAN 5 değişken:")
for ad, deger in katsayilar.head(5).items():
    print(f"         {ad:42s} {deger:+.3f}")

fig, eksenler = plt.subplots(1, 2, figsize=(15, 6))
eksen = eksenler[0]
ilk12 = perm_onem.head(12)[::-1]
eksen.barh(ilk12.index, ilk12.values * 100, color=YESIL, alpha=0.85)
eksen.set_title("Permütasyon önemi (test kümesi)")
eksen.set_xlabel("ROC-AUC kaybı (%)")
eksen.tick_params(axis="y", labelsize=8)
sadelestir(eksen)

eksen = eksenler[1]
uc_uca = pd.concat([katsayilar.head(6), katsayilar.tail(6)])
eksen.barh(uc_uca.index, uc_uca.values,
           color=[KIRMIZI if v > 0 else YESIL for v in uc_uca.values], alpha=0.9)
eksen.axvline(0, color=LACIVERT, lw=1.2)
eksen.set_title("Lojistik regresyon katsayıları\n(sağ = riski artırır, sol = azaltır)")
eksen.set_xlabel("Katsayı")
eksen.tick_params(axis="y", labelsize=8)
sadelestir(eksen)
grafik_kaydet(fig, "07_aciklanabilirlik.png")

# --- Tek bir başvuruyu açıklama
print("\n   [C] İKİ BAŞVURUNUN AÇIKLANMASI")
print("       Modelin en emin olduğu riskli ve en güvenli iki başvuruyu inceleyelim.")

for etiket, ornek in [("EN RİSKLİ BULUNAN", int(np.argmax(test_olasilik))),
                      ("EN GÜVENLİ BULUNAN", int(np.argmin(test_olasilik)))]:
    satir = df.loc[X_test.index[ornek]]
    print(f"""
       --- {etiket} BAŞVURU (test kümesi sırası: {ornek}) ---
         Hesap durumu   : {satir['hesap_durumu']}
         Kredi geçmişi  : {satir['kredi_gecmisi']}
         Birikim        : {satir['birikim']}
         Kredi tutarı   : {satir['kredi_tutari']:.0f} DM   ({satir['vade_ay']:.0f} ay vade)
         Aylık taksit   : {satir['aylik_taksit']:.0f} DM
         Yaş            : {satir['yas']:.0f}

         Modelin tahmini : {'RİSKLİ' if test_tahmin[ornek] == 1 else 'İYİ'}
         Risk olasılığı  : %{test_olasilik[ornek]*100:.1f}
         Gerçek durum    : {'RİSKLİ' if y_test.iloc[ornek] == 1 else 'İYİ'}
         Sonuç           : {'DOĞRU tahmin' if test_tahmin[ornek] == y_test.iloc[ornek] else 'YANLIŞ tahmin'}""")

if isinstance(EN_IYI_MODEL, SVC):
    print("""
       TEKNİK NOT: SVM'de predict() ile predict_proba() bazen çelişebilir.
       predict(), karar sınırının hangi tarafında kaldığınıza bakar;
       predict_proba() ise ayrı bir kalibrasyon adımıyla (Platt scaling)
       sonradan hesaplanır. Ayrıca class_weight="balanced" kullandığımız için
       karar sınırı %50 olasılık çizgisinden kaydırılmıştır. Bu yüzden
       "riskli" etiketi alan bir başvurunun olasılığı %50'nin altında
       görünebilir — hata değil, modelin çalışma biçiminin sonucu.""")

print("""
   NOT: Aynı analiz SHAP veya LIME ile de yapılabilir (derste 8. bölümde
   işlendi). Bu projede, ek bağımlılık gerektirmeyen ve scikit-learn'ün
   kendi içinde bulunan permütasyon önemi tercih edildi — sonuç yorumu
   benzer, kurulumu daha sağlam.
""")


# ======================================================= ÖZET
baslik("PROJE ÖZETİ")
print(f"""
   Veri seti          : UCI German Credit — 1.000 başvuru, 20 değişken
   Problem türü       : İkili sınıflandırma (iyi / riskli müşteri)
   Üretilen öznitelik : 5 adet (aylik_taksit, kredi_yas_orani, log_kredi_tutari,
                        toplam_yuk, yas_grubu)
   Öznitelik seçimi   : {X_train.shape[1]} sütundan {len(SECILEN)} sütuna indirildi
   Eğitilen model     : 5 adet
   Seçilen model      : {EN_IYI_AD} (GridSearchCV ile ayarlandı)

   TEST SONUÇLARI
     Accuracy  : %{test_accuracy:.2f}
     Precision : %{test_precision:.2f}
     Recall    : %{test_recall:.2f}
     F1-Score  : %{test_f1:.2f}
     ROC-AUC   : %{test_auc:.2f}

   Grafikler 'gorseller/' klasörüne kaydedildi.
""")
print("=" * 78)
print("  Çalışma tamamlandı.")
print("=" * 78)
