#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır. Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri
# içeren veri setini kullanarak Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId     : Müşteri numarası
# ServiceId  : Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId : Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate : Hizmetin satın alındığı tarih

# Veriyi Hazırlama

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv("DATASETS/armut_data.csv")
df = df_.copy()

df.head()
#   UserId  ServiceId  CategoryId           CreateDate
# 0   25446          4           5  2017-08-06 16:11:00
# 1   22948         48           5  2017-08-06 16:12:00
# 2   10618          0           8  2017-08-06 16:13:00
# 3    7256          9           4  2017-08-06 16:14:00
# 4   25446         48           5  2017-08-06 16:16:00

df.info()
# RangeIndex: 162523 entries, 0 to 162522
# Data columns (total 4 columns):
#  #   Column      Non-Null Count   Dtype
# ---  ------      --------------   -----
#  0   UserId      162523 non-null  int64
#  1   ServiceId   162523 non-null  int64
#  2   CategoryId  162523 non-null  int64
#  3   CreateDate  162523 non-null  object
# dtypes: int64(3), object(1)

df.shape
# (162523, 4)

df.isnull().sum()
# UserId        0
# ServiceId     0
# CategoryId    0
# CreateDate    0
# dtype: int64

df.describe().T
#               count          mean          std  min     25%      50%      75%      max
# UserId      162523.0  13089.803862  7325.816060  0.0  6953.0  13139.0  19396.0  25744.0
# ServiceId   162523.0     21.641140    13.774405  0.0    13.0     18.0     32.0     49.0
# CategoryId  162523.0      4.325917     3.129292  0.0     1.0      4.0      6.0     11.0

# ServisId her bir CategoryId özelinde farklı bir hizmeti temsil etmektedir.
# Bu sebeple 2 sini birleştirerek "Hizmet" i temsil edecek bir değişken oluşturalım.

# Çözüm 1:
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]

# Çözüm 2:
df[["ServiceId", "CategoryId"]] = df[["ServiceId", "CategoryId"]].astype(str)
df["Hizmet"] = ["_".join(col) for col in (df[["ServiceId", "CategoryId"]].values)]

#          Hizmet
# 0          4_5
# 1         48_5
# 2          0_8
# 3          9_4
# 4         48_5
#           ...
# 162518    25_0
# 162519     2_0
# 162520    31_6
# 162521    38_4
# 162522    47_7

# Veride herhangi bir sepet tanımı bulunmamaktadır. ARL (Association Rule Learning) yapabilmek için
# bir sepetimiz olmalıdır. Bu veride sepeti her bir müşterinin aylık aldığı hizmetler olarak tanımlayabiliriz.

# CreateDate değişkenini önce Object ten date sonra da yıl ve ay cinsinden olacak şekle çevirelim.
# Genel Date Çevrimlerinin Kısaltmaları
# %a : hafta gününün kısaltılmış adı
# %A : hafta gününün tam adı
# %b : ayın kısaltılmış adı
# %B : ayın tam adı
# %c : tam tarih, saat ve zaman bilgisi
# %d : sayı değerli bir karakter dizisi olarak gün
# %j : belli bir tarihin, yılın kaçıncı gününe denk geldiğini gösteren 1-366 arası bir sayı
# %m : sayı değerli bir karakter dizisi olarak ay
# %U : belli bir tarihin yılın kaçıncı haftasına geldiğini gösteren 0-53 arası bir sayı
# %y : yılın son iki rakamı
# %Y : yılın dört haneli tam hali
# %x : tam tarih bilgisi
# %X : tam saat bilgisi

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["Date"] = df["CreateDate"].dt.strftime("%Y-%m")
#             Date
# 0         2017-08
# 1         2017-08
# 2         2017-08
# 3         2017-08
# 4         2017-08
#            ...
# 162518    2018-08
# 162519    2018-08
# 162520    2018-08
# 162521    2018-08
# 162522    2018-08

# Sepetimizi oluşturalım.

# Çözüm 1:
df["SepetId"] = [str(row[0]) + "_" + str(row(5)) for row in df.values]

# Çözüm 2:
df[["UserId", "Date"]] = df[["UserId", "Date"]].astype(str)
df["SepetId"] = ["_".join(col) for col in (df[["UserId", "Date"]].values)]
#              SepetId
# 0         25446_2017-08
# 1         22948_2017-08
# 2         10618_2017-08
# 3          7256_2017-08
# 4         25446_2017-08
#               ...
# 162518    10591_2018-08
# 162519    10591_2018-08
# 162520    10591_2018-08
# 162521    12666_2018-08
# 162522    17497_2018-08

df.head()
#  UserId ServiceId CategoryId          CreateDate Hizmet     Date        SepetId
# 0  25446         4          5 2017-08-06 16:11:00    4_5  2017-08  25446_2017-08
# 1  22948        48          5 2017-08-06 16:12:00   48_5  2017-08  22948_2017-08
# 2  10618         0          8 2017-08-06 16:13:00    0_8  2017-08  10618_2017-08
# 3   7256         9          4 2017-08-06 16:14:00    9_4  2017-08   7256_2017-08
# 4  25446        48          5 2017-08-06 16:16:00   48_5  2017-08  25446_2017-08

# Sepetimizi oluşturduktan sonra artık Brliktelik Kuralı oluşturmasına geçebiliriz.

# Sepet ve Hizmet pivot tablomuzu oluşturalım.
# Sepet ve Hizmetleri binary cinsinden ifadeye çeviriyoruz.
# unstack   = Satırda bulunan hizmet dğeişkenini sütunlara geçirir. Pivotlaştırma işlemi yapar.
# fillna(0) = Pivot işleminden sonra oluşan NaN değerleri 0 a çevirir. (float olarak, 0.0 gibi)
# Son olarak applymap fonksiyonu ile satır ve sütunların hepsinde gezerek 0-1 düzenine getiriyoruz.

service_product_df = df.groupby(["SepetId", "Hizmet"])["Hizmet"].count().unstack().fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0)

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4  19_6  1_4  20_5  21_5  22_0  23_10  24_10  25_0  26_7  27_7  28_4  29_0  2_0  30_2  31_6  32_4  33_4  34_6  35_11  36_1  37_0  38_4  39_10  3_5  40_8  41_3  42_1  43_2  44_0  45_6  46_4  47_7  48_5  49_1  4_5  5_11  6_7  7_3  8_5  9_4
# SepetId
# 0_2017-08        0     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0     0     0     0     0     0    0     0     0     0     0     0      0     0     0     0      0    0     0     0     0     0     0     0     1     0     1     0    0     0    0    0    0    0
# 0_2017-09        0     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0     0     0     0     0     0    0     0     0     0     0     0      0     0     0     0      0    0     0     0     0     0     0     0     0     0     1     0    1     0    0    0    0    0
# 0_2018-01        0     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0     0     0     0     0     0    0     1     0     0     0     0      0     0     0     0      0    0     0     0     0     1     0     0     0     0     0     0    0     0    0    1    0    0
# 0_2018-04        0     0      0     0      0     1     0     0     0     0     0    0     0     0     0      0      0     0     0     0     0     0    0     1     0     0     0     0      0     0     0     0      0    0     0     0     0     1     0     0     0     0     0     0    0     0    0    0    0    0
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0     0     0     0     0     0    0     0     0     0     0     0      0     0     0     0      0    0     0     0     0     0     0     0     1     0     0     0    0     0    0    0    0    0
#             ...   ...    ...   ...    ...   ...   ...   ...   ...   ...   ...  ...   ...   ...   ...    ...    ...   ...   ...   ...   ...   ...  ...   ...   ...   ...   ...   ...    ...   ...   ...   ...    ...  ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...  ...   ...  ...  ...  ...  ...
# 99_2017-12       1     0      0     0      0     0     0     0     0     0     0    0     1     0     0      0      0     0     0     0     0     0    0     0     0     0     0     0      0     0     0     0      0    0     0     0     0     0     0     0     0     0     0     0    0     0    0    0    0    0
# 99_2018-01       1     0      0     0      0     0     0     0     0     0     0    0     1     0     0      0      0     0     0     0     0     0    0     0     0     0     0     0      0     0     0     0      0    0     0     0     0     0     0     0     0     0     0     0    0     0    0    0    0    0
# 99_2018-02       1     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0     0     0     0     0     0    0     0     0     0     0     0      0     0     0     0      0    0     0     0     0     0     0     0     0     0     0     0    0     0    0    0    0    0
# 9_2018-03        0     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0     1     0     0     0     0    0     0     0     0     0     0      0     0     0     0      0    0     0     0     0     0     0     0     0     0     0     0    0     0    0    0    0    0
# 9_2018-04        0     0      0     0      0     0     0     0     0     0     1    0     0     0     0      0      0     1     0     0     0     0    0     0     0     0     0     0      0     0     0     0      0    0     0     0     0     0     0     0     0     0     0     0    0     0    0    0    0    0

# Birliktelik Kuralı (Apriori Algoritması)
frequent_services = (apriori(service_product_df, min_support=0.01, use_colnames=True))
# metric değişkendir. confidence vb. gibi yazılabilir.
rules = association_rules(frequent_services, metric="support", min_threshold=0.01)
rules.head()

# support             = 2  Hizmetin birlikte alınma frekansı
# antecedent support  = 1. Hizmetin Tek başına alınma olasılı
# consequent support  = 2. Hizmetin Tek başına alınma olasılı
# antecedents         = 1. Hizmet
# consequents         = 2. Hizmet
# confidence          = 1  Hizmet alındığında 2. Hizmetin alınma olasılığı
# lift                = 1  Hizmet alındığında 2. Hizmetin alınma olasılığının kaç kat artacağının belirtir.
# leverage            = lift benzeridir. Ancak support bağlı olduğu için yanlıdır.
# conviction          = 2. Hizmet olmadan 1. Hizmetin beklenen frakansı ya da
                        # 1. Hizmet olmadan 2. Hizmetin beklenen frakansı diyebiliriz.

#   antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction
# 0     (13_11)       (2_0)            0.056627            0.130286  0.012819    0.226382  1.737574  0.005442    1.124216
# 1       (2_0)     (13_11)            0.130286            0.056627  0.012819    0.098394  1.737574  0.005442    1.046325
# 2       (2_0)      (15_1)            0.130286            0.120963  0.033951    0.260588  2.154278  0.018191    1.188833
# 3      (15_1)       (2_0)            0.120963            0.130286  0.033951    0.280673  2.154278  0.018191    1.209066
# 4      (33_4)      (15_1)            0.027310            0.120963  0.011233    0.411311  3.400299  0.007929    1.493211

# Öneri yapacak olan Fonksiyonumuzu kuralım.

def arl_recomender(rules_df, product_id, rec_count=1):
    # Verinin lift e göre büyükten küçüğe sıralama
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recomender(rules, "2_0", 1)
# ['22_0'] Hizmetini Önerdi.
arl_recomender(rules, "2_0", 2)
# ['22_0', '25_0'] Hizmetlerini önerdi.
arl_recomender(rules, "2_0", 3)
# ['22_0', '25_0', '15_1'] Hizmetlerini önerdi.

# Not = Önerilen hizmet sayısı arttıkça birlikte görülme olasılığı düşmektedir.