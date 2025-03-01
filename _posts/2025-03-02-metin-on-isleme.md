---
title: 01-Metin Ön İşleme
layout: post
categories: [Tutorial, NLP Tutorial]
tags: [tutorial]     # TAG names should always be lowercase
author: 01
---

# 📌 Metin Ön İşleme: Neden Önemlidir?

Doğal Dil İşleme (NLP) projelerinde ham metin verisi, çoğu zaman doğrudan analiz veya modelleme için kullanılamaz. Metinler; düzensiz karakterler, noktalama işaretleri, gereksiz kelimeler ve biçimsel farklılıklar içerebilir. Bu nedenle, metinlerin doğru ve etkili bir şekilde işlenebilmesi için ön işleme adımları uygulanmalıdır.

Metin ön işleme, veriyi daha temiz, tutarlı ve işlenebilir hale getirmek için yapılan bir dizi dönüşüm sürecini kapsar. Bu süreç, kullanılan modele ve uygulama alanına bağlı olarak değişebilir ancak genel olarak şu nedenlerle kritik bir öneme sahiptir:

✅ **Gürültüyü Azaltma**: Gereksiz karakterler, semboller ve durak (stop) kelimelerinin kaldırılması, veriyi daha anlamlı hale getirir.  
✅ **Veri Tutarlılığını Artırma**: Farklı biçimlerde yazılmış kelimeleri normalize etmek, metnin daha tutarlı olmasını sağlar.  
✅ **Hesaplama Maliyetini Azaltma**: Gereksiz veriyi eleyerek modelin daha verimli çalışmasına yardımcı olur.  
✅ **Genelleştirme Yeteneğini Artırma**: Kelime köklerine indirgeme (stemming/lemmatization) gibi işlemler, modelin farklı bağlamlarda da iyi performans göstermesini sağlar.  

---

## 🔍 Metin Ön İşleme Aşamaları  

Metin ön işleme süreci, çeşitli dönüşümleri içeren birden fazla adımdan oluşur:

1️⃣ **Büyük/Küçük Harf Dönüşümü (Lowercasing)**: Metindeki tüm harflerin küçük harfe çevrilmesi.  
2️⃣ **Noktalama İşaretlerinin Kaldırılması (Removing Punctuation)**: Noktalama işaretlerinin silinmesi veya gerektiğinde değiştirilmesi.  
3️⃣ **Sayıların Kaldırılması (Removing Numbers)**: Sayıların metinden çıkarılması (bazı durumlarda korunabilir).  
4️⃣ **Durak Kelimelerin Kaldırılması (Removing Stopwords)**: "ve", "bu", "bir" gibi bilgi taşımayan yaygın kelimelerin çıkarılması.  
5️⃣ **Özel Karakterlerin ve Emojilerin Kaldırılması (Removing Special Characters & Emojis)**: Metindeki özel sembollerin ve emojilerin temizlenmesi.  
6️⃣ **Metin Normalizasyonu (Text Normalization)**: Kelimelerin standart bir forma dönüştürülmesi (örneğin, "olmıyacak" → "olmayacak", "tlfn" → "telefon").  
7️⃣ **Kök veya Gövdeye İndirgeme (Stemming & Lemmatization)**: Kelimeleri kök haline getirme (stemming) veya köküne en yakın hâle getirme (lemmatization).  
8️⃣ **Kelime Tokenizasyonu (Tokenization)**: Metni kelime veya cümlelere ayırma.   

```python
import os
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("mustfkeskin/turkish-movie-sentiment-analysis-dataset")
dataset_name = os.listdir(path)[0]
print("Path to dataset files:", path)
print("Dataset name:", dataset_name)
````
```python
df = pd.read_csv(os.path.join(path, dataset_name))
print(df["comment"][4])
df.head()
```
---

## 📌 1. Aşama: Lowercasing (Küçük Harfe Çevirme)

**Neden Önemlidir?**  
Metinlerde büyük/küçük harf duyarlılığı nedeniyle aynı kelimeler farklı şekilde yorumlanabilir.  
Örneğin, "Film" ve "film" kelimeleri aslında aynıdır, ancak büyük harf farkı nedeniyle farklı olarak algılanabilir.  
Bunu önlemek için tüm metni küçük harfe çevirmeliyiz.

### 🔹 Örnek:
- **Girdi:** `"Bu Film HARİKAYDI!"`
- **Çıktı:** `"bu film harikaydı!"`

Aşağıdaki kod bloğu, veri setindeki belirli bir sütunu tamamen küçük harfe çevirerek normalize eder.

```python
text_column = "comment" 

print("GİRDİ:", df[text_column][4])

df[text_column] = df[text_column].str.lower()

print("ÇIKTI:", df[text_column][4])
```

---

## 📝 2. Aşama: Noktalama İşaretlerini Temizleme

### **Neden Önemlidir?**
Metinlerdeki noktalama işaretleri, dilin yapısal öğeleri olup, modelin anlam çıkarma ve analiz yapma konusunda yanıltıcı olabilir. Örneğin, bir cümledeki noktalama işaretleri, genellikle anlam taşımayan karakterlerdir. Noktalama işaretlerini temizlemek, metni daha uygun hale getirebilir.

### 🔹 **Örnek:**
- **Girdi:** `"Film harikaydı, ancak biraz uzun."`
- **Çıktı:** `"film harikaydi ancak biraz uzun"`

### 🛠️ **Noktalama İşaretlerini Temizleme Adımları:**
Bu aşamada, metinden tüm noktalama işaretlerini kaldıracağız. Bunun için **regex** kullanarak, metin içindeki noktalama işaretlerini temizleyeceğiz.

```python
import string

string.punctuation
print("GİRDİ:", df[text_column][4])

df[text_column] = df[text_column].str.replace(f"[{string.punctuation}]", "", regex=True)

print("ÇIKTI:", df[text_column][4])
``` 

---

## 📝 3. Aşama: Sayıları Temizleme

### **Neden Önemlidir?**
Metinlerde bulunan sayılar, genellikle model için anlam taşımaz. Eğer verisetinde belirli bir sayısal bilgi yoksa ya da sayılar analizi etkilemeyecekse, bunların metinden çıkarılması gerekebilir. Özellikle duygu analizi gibi görevlerde sayılar, anlamlı bir bilgi taşımaz. 

### 🔹 **Örnek:**
- **Girdi:** `"Bu film 2021 yılında vizyona girdi."`
- **Çıktı:** `"bu film yılında vizyona girdi"`

### 🛠️ **Sayıları Temizleme Adımları:**
Bu aşamada, metindeki sayıları kaldıracağız. Bunun için **regex** kullanarak sayıları temizleyeceğiz.

```python
print("GİRDİ:", df[text_column][4])
df[text_column] = df[text_column].str.replace(r'\d+', '', regex=True)

print("ÇIKTI:", df[text_column][4])
```

---

## 📝 4. Aşama: Stopwords (Anlamsız Kelimeleri) Temizleme

### **Neden Önemlidir?**
**Stopwords**, dilin temel yapı taşları olan ve genellikle anlam taşımayan, çok sık kullanılan kelimelerdir. Bu kelimeler, modelin anlamlı bilgi öğrenmesini engelleyebilir. Örneğin, "ve", "bir", "ile" gibi kelimeler bir cümlenin anlamını değiştirmez, ancak metin analizinde yer alması modelin yanlış öğrenmesine yol açabilir. Bu yüzden stopwords'leri metinden çıkararak metni sadeleştirebiliriz.

### 🔹 **Örnek:**
- **Girdi:** `"Film çok güzeldi ve oyunculuk harikaydı."`
- **Çıktı:** `"film güzeldi oyunculuk harikaydı"`

### 🛠️ **Stopwords Temizleme Adımları:**
Bu adımda, Türkçe stopwords listesini kullanarak metin içindeki anlamsız kelimeleri temizleyeceğiz.

```python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('turkish'))
print("STOPWORDS:", stop_words)

print("GİRDİ:", df[text_column][4])
df[text_column] = df[text_column].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
print("ÇIKTI:", df[text_column][4])
```

---

## 📝 5. Aşama: Özel Karakterlerin ve Emojilerin Kaldırılması

### **Neden Önemlidir?**
Metinlerdeki özel karakterler ve emojiler, modelin doğru şekilde öğrenmesi için genellikle gereksizdir. Özellikle duygu analizi veya metin sınıflandırma gibi görevlerde, bu semboller anlam taşımaz ve metnin doğasına zarar verebilir. Bu yüzden, emojiler ve özel karakterler, metin işleme sürecinde temizlenmesi gereken öğelerdir.

### 🔹 **Örnek:**
- **Girdi:** `"Film harikaydı! 😍🎬 Çok eğlenceliydi... :)"`
- **Çıktı:** `"film harikaydi cok eglenceliydi"`

### 🛠️ **Özel Karakterlerin ve Emojilerin Kaldırılması Adımları:**
Bu adımda, metindeki tüm özel karakterleri ve emojileri kaldıracağız. Bunun için regex kullanarak bu karakterleri temizleyeceğiz.

```python
import re

def convert_turkish_chars(text):
    turkish_chars = {
        'ç': 'c', 'ı': 'i', 'ğ': 'g', 'ş': 's', 'ü': 'u', 'ö': 'o', 'Ç': 'C', 'İ': 'I', 'Ğ': 'G', 'Ş': 'S', 'Ü': 'U', 'Ö': 'O'
    }
    for turkish_char, english_char in turkish_chars.items():
        text = text.replace(turkish_char, english_char)
    return text


def remove_special_characters_and_emojis(text):
    text = convert_turkish_chars(text)
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'[^\x00-\x7F]+', '', text)  
    return text


print("GİRDİ:", df[text_column][588])
df[text_column] = df[text_column].apply(remove_special_characters_and_emojis)
print("ÇIKTI:", df[text_column][588])
```

---

## 📝 6. Aşama: Metin Normalizasyonu (Text Normalization)

### **Neden Önemlidir?**
Metin normalizasyonu, metinlerdeki varyasyonları ortadan kaldırarak daha tutarlı ve temiz veriler elde etmemizi sağlar. Bu işlem, yazım hataları, kısaltmalar ve dildeki diğer tutarsızlıkların düzeltilmesine yardımcı olur. Modelin daha etkili bir şekilde öğrenmesi için metnin daha anlaşılır ve düzenli hale getirilmesi gereklidir.

### 🔹 **Örnek:**
- **Girdi:** `"Yazılım olmyacak, ama tlfn alcam!"`
- **Çıktı:** `"yazilim olmayacak ama telefon alacagim"`

### 🛠️ **Metin Normalizasyonu Adımları:**
Bu adımda, yazım hatalarını düzeltecek ve bazı yaygın kısaltmaları genişleterek daha standart bir forma dönüştüreceğiz. Bunun için bir kelime eşlemesi (dictionary) kullanabiliriz.

```python

normalization_dict = {
    'tmm': 'tamam',
    'gzl': 'guzel'
}

def normalize_text(text):
    text = text.lower() 
    for word, normalized_word in normalization_dict.items():
        text = re.sub(r'\b' + word + r'\b', normalized_word, text)
    return text

print("GİRDİ:", df[text_column][2232])

df['comment'] = df['comment'].apply(normalize_text)

print("ÇIKTI:", df[text_column][2232])
```

---

### 📝 **7. Aşama: Kök veya Gövdeye İndirgeme (Stemming & Lemmatization)**

**Neden Önemlidir?**  
Metindeki kelimeleri köklerine indirgeme (stemming) veya köklerine en yakın hale getirme (lemmatization), kelimelerin farklı biçimlerini tek bir temsil ile ifade eder. Bu, modelin daha anlamlı bir şekilde çalışmasına yardımcı olur. Örneğin, "yazıyorum", "yazdı" ve "yazmak" kelimeleri köklerine indirgenerek hepsi "yaz" olarak işlenebilir. Bu, dilin çeşitli türevlerinin anlamını birleştirir ve modelin öğrenmesini iyileştirir.

---

**Stemming ve Lemmatization Arasındaki Farklar:**

- **Stemming**: Kelimenin köküne indirgeme işlemi yapar, ancak bazı kelimeleri yanlış köklerle eşleştirebilir. Örneğin, "yazmak" kelimesi "yaz" olarak indirgenir.
  
- **Lemmatization**: Daha gelişmiş bir işlemdir. Kelimenin doğru kök biçimini bulur. "yazmak" → "yaz", "güzel" → "güzel" gibi. Bu işlem için bir kelime çözümleyici (lemmatizer) kullanılır.

### 1️⃣ **Stemming Örneği:**

```python

from jpype import startJVM, JClass, shutdownJVM
ZEMBEREK_PATH = "./zemberek-full.jar" 

startJVM(classpath=ZEMBEREK_PATH)

TurkishMorphology = JClass("zemberek.morphology.TurkishMorphology")
morphology = TurkishMorphology.createWithDefaults()

def stem(word):
    analysis = morphology.analyze(word)
    results = [str(result.getStem()) for result in analysis]
    return results[0] if results else word

print("GİRDİ:", df[text_column][10])
word_list = df[text_column][10].split()
print("ÇIKTI:", [stem(word) for word in word_list])

shutdownJVM()
```

### 2️⃣ **Lemmatization Örneği:**

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatization(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


print("GİRDİ:", df[text_column][2])

print("ÇIKTI:", lemmatization(df[text_column][2]))
```

## 📝 8. Aşama: Kelime Tokenizasyonu (Tokenization)

### 🚀 Neden Önemlidir?
Kelime tokenizasyonu, metni daha küçük birimlere (kelime veya cümlelere) ayırma işlemdir. Doğal dil işleme (NLP) süreçlerinde veriyi daha anlamlı hale getirmek için temel bir adımdır.

**Örneğin:**

- **Kelime Bazlı Tokenizasyon:**  
  *"Bugün hava çok güzel!"*  
  Tokenizasyon işlemi: `["Bugün", "hava", "çok", "güzel", "!"]`

- **Cümle Bazlı Tokenizasyon:**  
  *"Bugün hava çok güzel! Dışarı çıkalım mı?"*  
  Tokenizasyon işlemi: `["Bugün hava çok güzel!", "Dışarı çıkalım mı?"]`

### 🛠 Tokenizasyon Türleri
1. **Kelime Bazlı Tokenizasyon:** Cümleyi kelimelere böler.
2. **Cümle Bazlı Tokenizasyon:** Metni cümlelere böler.
3. **Alt Kelime (Subword) Tokenizasyonu:** Özel karakterleri ve ekleri de hesaba katarak parçalama yapar.

```python
from trtokenizer.tr_tokenizer import SentenceTokenizer, WordTokenizer


sentence_tokenizer = SentenceTokenizer()

print("GİRDİ:", df[text_column][0])

print("ÇIKTI:", sentence_tokenizer.tokenize(df[text_column][0]))

word_tokenizer = WordTokenizer()

print("ÇIKTI:", word_tokenizer.tokenize(df[text_column][0]))
```

