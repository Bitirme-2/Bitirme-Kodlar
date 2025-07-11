# Yapay Zeka Destekli Akademik Arama Motoru

Bu proje, Gebze Teknik Üniversitesi lisans bitirme projesi kapsamında geliştirilmiş, anahtar kelimeler, proje özetleri veya belirli bir çağrıya göre en ilgili projeleri, proje çağrılarını ve uzman akademisyenleri tespit edebilen yenilikçi bir arama motorudur. Sistem, Doğal Dil İşleme (NLP) ve Bilgi Grafiği (Knowledge Graph) teknolojilerinden yararlanarak semantik arama ve ilişki keşfi yetenekleri sunar.

## 🎯 Projenin Amacı

Temel amaç, geleneksel anahtar kelime eşleşmesinin ötesine geçerek, anlamsal bağlama dayalı en doğru sonuçları sunan bir arama motoru geliştirmektir. Araştırma verileri (akademisyenler, projeler, makaleler vb.) arasındaki karmaşık ilişkiler bir bilgi grafiği olarak modellenmiş ve bu yapı üzerinden grafik madenciliği algoritmaları ile gizli desenlerin ve eğilimlerin keşfedilmesi hedeflenmiştir.

## ✨ Özellikler

Proje, kullanıcıların akademik veriler arasında verimli bir şekilde gezinmesini sağlayan zengin bir özellik setine sahiptir:

* **Akıllı Semantik Arama:** Girilen anahtar kelimeler veya proje özetleri üzerinden anlamsal benzerlik algoritmaları kullanarak ilgili projeleri, çağrıları ve akademisyenleri bulur.
* **Esnek Model Seçimi:** Kullanıcılara arama işlemi için farklı yapay zeka modelleri (Sentence Transformer ve PyKEEN tabanlı Bilgi Grafiği) arasında seçim yapma imkanı sunar.
* **Veri Yönetimi:**
    * **Makale Yükleme:** Sisteme manuel olarak veya bir URL üzerinden akademik makalelerin (PDF formatında) yüklenmesine olanak tanır. Sistem, PDF'lerden tam metin içeriğini çıkarabilir ve IEEE, arXiv gibi platformlardan meta veri çekebilir.
    * **Çağrı Ekleme:** Yeni proje çağrılarının başlık, açıklama ve son başvuru tarihi gibi bilgilerle sisteme eklenebilmesini sağlar.
    * **Veri Listeleme ve Silme:** Sistemdeki akademisyen, proje ve çağrı verilerinin listelenmesi ve istenmeyen kayıtların (ilişkili verilerle birlikte) kolayca silinmesi mümkündür.
* **İlişkili İçerik Keşfi:** Projelerle ilişkili akademisyenlerin ve akademisyenlerle ilişkili projelerin otomatik olarak bulunup görüntülenmesini sağlar.
* **Sistem Sağlığı İzleme:** Veritabanı bağlantısı ve aktif yapay zeka modelinin durumu gibi sistem bileşenlerinin gerçek zamanlı sağlık durumunu gösteren bir arayüz sunar.
* **PDF İndirme:** Sisteme yüklenmiş olan PDF dokümanlarının indirilmesine olanak tanır.

## 🛠️ Kullanılan Teknolojiler

Projenin geliştirilmesinde aşağıdaki teknolojiler ve kütüphaneler kullanılmıştır:

* **Backend:** Flask
* **Veritabanı:** Neo4j (Grafik Veritabanı)
* **Doğal Dil İşleme & Gömme Modelleri:**
    * Sentence-Transformers (`paraphrase-multilingual-mpnet-base-v2`, `all-MiniLM-L12-v2`)
    * PyKEEN (TransE modeli ile Bilgi Grafiği Gömülmesi)
* **Veri İşleme ve Analiz:**
    * Pandas, NumPy
    * Scikit-learn
* **PDF İşleme:** PyMuPDF
* **Web Veri Çıkarımı:** Requests, BeautifulSoup, Selenium

## 🚀 Kurulum ve Çalıştırma

Projenin yerel makinede çalıştırılması için aşağıdaki adımları izleyin:

1.  **Depoyu Klonlama:**
    ```bash
    git clone [https://github.com/kullanici-adi/bitirme-kodlar.git](https://github.com/kullanici-adi/bitirme-kodlar.git)
    cd bitirme-kodlar
    ```

2.  **Gerekli Paketlerin Yüklenmesi:**
    `requirements.txt` dosyasında listelenen tüm bağımlılıkları yükleyin.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Neo4j Veritabanı Kurulumu:**
    * Neo4j Desktop'ı indirip kurun.
    * Yeni bir proje ve veritabanı oluşturun.
    * `app.py` dosyasındaki veritabanı bağlantı bilgilerinizi (`"bolt://localhost:7687"`, kullanıcı adı ve şifre) kendi Neo4j bilgilerinizle güncelleyin.

4.  **Uygulamayı Başlatma:**
    Aşağıdaki komut ile Flask uygulamasını başlatın.
    ```bash
    python app.py
    ```

5.  **Arayüze Erişim:**
    Web tarayıcınızı açın ve `http://127.0.0.1:5001` adresine gidin.

## 🗂️ Proje Yapısı


.
├── app.py                  # Ana Flask uygulaması, tüm backend mantığı
├── train_pykeen.py         # PyKEEN modelini eğitmek için script
├── script2.py              # Modellerin performansını değerlendirmek için script
├── requirements.txt        # Proje bağımlılıkları
├── templates/              # HTML şablonları
│   ├── index.html
│   ├── upload.html
│   ├── add_call.html
│   ├── health.html
│   └── list.html
├── uploads/                # Yüklenen PDF'lerin saklandığı klasör
└── pykeen_model/           # Eğitilmiş PyKEEN modelinin saklandığı klasör


## 🧠 Modeller

Sistem, arama işlemleri için değiştirilebilir üç farklı model kullanır:

1.  **Sentence Transformer (Multilingual):** `paraphrase-multilingual-mpnet-base-v2` modelini kullanarak çok dilli metinler için anlamsal gömmeler (embeddings) oluşturur.
2.  **Sentence Transformer (MiniLM):** `all-MiniLM-L12-v2` modelini kullanır. Daha hızlıdır ancak genellikle İngilizce metinlerde daha iyi performans gösterir.
3.  **PyKEEN (Bilgi Grafiği Modeli):** Neo4j'deki verilerden (Akademisyen-Proje-Yayın ilişkileri) oluşturulan bir bilgi grafiğini `TransE` algoritmasıyla eğitir. Bu model, veriler arasındaki ilişkileri keşfederek arama yapar.

## 🧑‍💻 Katkıda Bulunanlar

* Yasir Şekerci
* Abdullah Enes Patır
* Feridun Taha Açıkyürek
