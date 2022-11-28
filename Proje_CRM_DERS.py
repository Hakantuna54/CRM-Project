import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format",
              lambda x: "%.5f" % x)  # 0 dan sonra kaç basamk olacağını gösteriyor. , den sonra 5 tane sayı olabilir.

############Bu satırı 1 kere çalıştır.###################################################################
df_ = pd.read_excel(r"C:\Users\hakan\PycharmProjects\CRM\online_retail_II.xlsx", sheet_name="Year 2009-2010")
###################################################################################


df = df_.copy()
df.head()
df.describe().T
df.isnull().sum()
# df = df[df["Invoice"].str.contains("C", na=False)]#içinde c barındıranları getir
df = df[~df["Invoice"].str.contains("C", na=False)]  # içinde c barındırmayanları getir
df.describe().T
df = df[(df["Quantity"] > 0)]
df.dropna(inplace=True)
df["TotalPrice"] = df["Price"] * df["Quantity"]
cltvc_c = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),  # ınvoicea lambda işlemini yap diyoruz.
                                         "Quantity": lambda x: x.sum(),
                                         "TotalPrice": lambda x: x.sum()})
cltvc_c.columns = ["total_transaction", "total_unit", "total_price"]
######################################
# 2.Ortalama Sipariş Değeri Hesaplama:(total price / total transaction)
######################################
cltvc_c.head()  # total price ve total transaction cltvc_c nin içinde şuan
cltvc_c["avarage_order_value"] = cltvc_c["total_price"] / cltvc_c["total_transaction"]
# total price / total_transaction ile avarage
# order valueyi bulduk. ve cltvc_c["avarage_order_value"] değişkeni olarak tuttuk.

###########################################
# 3.Satın alma sıklığı(purchase frequency):(total_transaction / total_number_of_customer)
###########################################
cltvc_c.head()
cltvc_c["total_transaction"]
cltvc_c["purchase_frequency"] = cltvc_c["total_transaction"] / cltvc_c.shape[0]

cltvc_c.shape[0]  # customer numbera ulaşmak için tuttuğumuz değişkenin 0.elemanını çağırıyoruz. Çünkü 0.değişken

###########################################
# 4. Repeat Rate, Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
###########################################
repeat_rate = cltvc_c[cltvc_c["total_transaction"] > 1].shape[0] / cltvc_c.shape[0]
# 1den fazla alışveriş yapan müşteriler geldi. Shape ile boyuta eriştik ve 0.elemanı
# 1 kere alışveriş yapan müşteri sayısını veriyor.
churn_rate = 1 - repeat_rate  # churn_rate 1- repeat_rate ile bulunur.

###########################################
# 5. profit margin(profit_margin = total_price * 0.10)
###########################################
cltvc_c["profit_margin"] = cltvc_c["total_price"] * 0.10

###########################################
# 6. müşteri değeri(customer_value = avarage_order_value * purchase_frequency)
###########################################
cltvc_c["customer_value"] = cltvc_c["avarage_order_value"] * cltvc_c["purchase_frequency"]

###########################################
# 7. müşteri yaşam boyu değeri(CLTV = (customer_value / churn_rate) * profit_margin)
###########################################
cltvc_c["cltv"] = (cltvc_c["customer_value"] / churn_rate) * cltvc_c["profit_margin"]
# Bir dataframe oluşturduk ki bu müşterilerimizin tekil nihai olarak oluşturduğumuz veri setiydi.
cltvc_c.sort_values(by="cltv",ascending=False).head  # cltv_c yi sort ettik(sıraladık), ardından büyükten küçüğe sırala dedik FAlse
# diyerek.

###########################################
# 8. Segmentlerin Oluşturulması:
###########################################
cltvc_c.sort_values(by="cltv", ascending=False).tail()
cltvc_c["segment"] = pd.qcut(cltvc_c["cltv"], 4, labels=["D", "C", "B", "A"])
cltvc_c.sort_values(by="cltv", ascending=False).head()
cltvc_c.groupby("segment").agg({"count", "mean", "sum"})



###########################################
# 9. BONUS: Tüm işlemlerin fonksiyonlaştırılması:
###########################################
def create_cltv_c(dataframe, profit=0.10): #bir argümanı var dataframe, profitte biçimlendirilebilir
    #Veriyi Hazırlama:
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]#c yi barındıranları çıkar
    dataframe = dataframe[(dataframe["Quantity"] > 0)] #Quantity > 0 olanları getir.
    dataframe.dropna(inplace=True) #naleri çıkar.
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]#total priceleri hesapla
    cltvc_c = dataframe.groupby("Customer ID").agg({"Invoice":lambda x:x.nunique(),
                                                    "Quantity":lambda x:x.sum(),
                                                    "TotalPrice":lambda x:x.sum()}) #customer ıdlere göre group bya al.
    cltvc_c.columns = ["total_transaction", "total_unit", "total_price"] #bu şekilde isimlendir
    #avg_order_value
    cltvc_c["avg_order_value"] = cltvc_c["total_transaction", "total_unit", "total_price"]

    #purchase_frequency
    cltvc_c["purchase_frequency"] = cltvc_c["total_transaction"] / cltvc_c.shape[0]

    #repeat rate & churn rate
    repeat_rate = cltvc_c[cltvc_c.total_transaction > 1].shape[0] / cltvc_c.shape[0]
    churn_rate = 1 - repeat_rate

    #profit_margin
    cltvc_c["profit_margin"] = cltvc_c["total_price"] * profit

    #customer value
    cltvc_c["customer_value"] = (cltvc_c["avg_order_value"] * cltvc_c["purchase_frequency"])

    #customer lifetime value
    cltvc_c["cltv"] = (cltvc_c["customer_value"] / churn_rate) * cltvc_c["profit_margin"]
    # Segment
    cltvc_c["segment"] = pd.qcut(cltvc_c["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltvc_c
df = df_.copy()
clv = create_cltv_c(df)


##########################################################
#Müşteri Yaşam Boyu Değeri Tahmini:
##########################################################

##########################################################
#Zaman projeksiyonlu olasılıksal lifetime value tahmini:
##########################################################

#Satın alma başına ortalama kazanç * satın alma sayısı = müşteri yaşam boyu değeri
#CLTV = (customer value / churn rate) * profit margin
#customer value = purchase frequency * avarage order value
#CLTV = expected number of transaciton * expected avarage profit
#cltv = bg/nbd model * gamma gamma submodel bu 2 modelle cltvyi elde edeceğiz.


############################################################
#BG/NDB (beta geometrik / negatif binomal distribiton ile expected number of transaction
############################################################
#expected : bir rassal değişkenin beklenen değerini ifade eder rasssal değişkenin beklenen değeri rassal değişkenin ortalaması
#rasssal değişken değerlerini bir deneyin sonuçlarından alan değere rassal değişken denir
#bg/ndb : ölene kadar sayın al da denebilir. buy till you die
#bg/ndb modeli, expected number of transaction için iki süreci olasılıksal olarak modller
#transaction procces buy (satın alma işlemi süreci) + dropout proses(markayı terk etme)
#transaciton process buy: alive olduğu sürece, belirli bir zaman periyodunda, bir müşteri tarafından gerçekleştirilecek işlem
#sayısı transaction rate parametresi poisson dağılır. Yani bir müşteri alive olduğu sürece kendi transactionu etrafında rastgele
#satın alma yapmaya devam edecektir. Örneğin:mehmet bey bir alışveriş yaptı 5br sonra 10 sonra 15 vs. mehmet beyin bir satın alma davranışı var
#bu davranışını kendi satın alma davranışı etrafında devam ettirir.
#trarnsaction rateler her müşteriye göre değişir ve tüm kitle için gama dağılır.(r,a) gama bir olasılık dağılımıdır.
#satın alma alışkanlıkları tğm müşterilere göre farklı gerçekleşir.


#dropout proses: her bir müşterinin p olasılığıyla dropout olma olasılığı vardır. yani churn olma satın almayı bırakmadır.
#bir müşteri alışveriş yaptıktan sonra belirli bir olasılıkla drop olur.
#dropout rateler her bir müşteriye göre değişir ve tüm kitle için beta dağılır.
#şöyleki işlem oranı gama dağılır drop outlar beta dağılır.



####################################################
#Gama Gama Submodel:
####################################################
#Bir müşterinin işlem başına ne kadar kar getirebileceğini tahmin etmek için kullanılır.
#bir müşterinin işlemlerinin parasal değeri (monetary) transaction valueler etrafında rastgle dağılır.
#ortalama transaction value, zaman içinde kullanıcılar arasında değişebilir fakat tek bir kullanıcı içermez.
#örneğn mehmet bey bir alışveriş yaptı ve bir parasal değer bıraktı. Daha sonra baika bir alışveriş yaptı. 100-300-200 tl lik işlemler olsun
#bu işlemlerin ortalmasını alırız. ve mehmet bey bu parasal değer etrafında ortalama bir getiri bırakmaya devam eder.
#kullanıcılar arasında değişebilir ama kullanıcıda değişmez. ortalama trasaction value tüm müşteriler arasında gamma dağılır.
#******CRM açısından cltv aşırı önemlidir******


##############################################
#BG/NBD  ve gama gama ile cltv tahmini:
##############################################
#1.Verinin hazırlanması
#2.bg/nbd modeli ile expected number of transaction
#3.Gama-gama modeli ile expected avarage profit
#4.bg/nbd ve gama gama modeli ile cltv'nin hesaplanması
#5.cltv ye göre segmentlerin oluşturulması
#6.çalışmanın fonksiyonlaştırılması


##############################################
#1.verinin hazırlanması (data-preperation)
##############################################

#gerekli kütüphane ve fonksiyonlar
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x) #virgülden sonra 4.basamağa kadar göster
from sklearn.preprocessing MinMaxScaler #0-100 arasındaki değerlere çekmek istersek kullanırız.



def outlier_threshold(dataframe, variable): #önce aykırı değerleri quantile yöntemiyle tesepit edeceğiz.
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 * 1.5 * interquantile_range #üst sınır için 1.5 kat yukarıda olan değerler benim için aykırı değerdir
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_threshold(dataframe, variable): #vir dataframe ve değişken ile çağırdığımızda
    low_limit, up_limit = outlier_threshold(dataframe,variable)#yukarıdan fonksiyonu çağırıp üst ve alt değerleri ister
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#############################################################
#Verinin okunması:
#############################################################
#veri setini yukarıdan okuttumdf_ = pd.read_excel("C:\Users\hakan\PycharmProjects\CRM\online_retail_II.xlsx", sheet_name= "Year 2009-2010")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

#############################################################
#Verinin ön işleme:
#############################################################
df.dropna(inplace = True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
replace_with_threshold(df, "Quantity")
replace_with_threshold(df, "Price")
df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)#analiz tarihi olarak bu tarihden 1-2 gün sonrayı alıyoruz.


#############################################################
#Lifetime Veri Yapısının Hazırlanması
##############################################################
#recency : son satın alma üzerinden geçen zaman. Haftalık(kullanıcı özelinde)
#T : müşterinin yaşı(haftalık)(analiz tarihinden ne kadar süre önce alışveriş yapılmış)
#frequency : tekrar eden toplam satın alma sayısı(frequency>1)
#monetary: satın alma başına ortalama kazanç. eskiden toplam kazançtı
cltvc_df = df.groupby("Customer ID").agg({"InvoiceDate" : [lambda InvoiceDate : (InvoiceDate.max() - InvoiceDate.min()).days,#son alışveriş tarihinden ilk alışveriş tarihini çıkar ve gün cinsinden ifade et
                                                           lambda InvoiceDate : (today_date - InvoiceDate.min()).days], #müşteri yaşını hesaplıyoruz.
                                          "Invoice" : lambda Invoice: Invoice.nunique(),
                                          "TotalPrice":lambda TotalPrice:TotalPrice.sum()})
cltvc_df.columns = cltvc_df.columns.droplevel(0)

cltvc_df.columns = ["recency", "T", "frequency", "monetary"]
cltvc_df.describe().T
cltvc_df = cltvc_df[(cltvc_df["frequency"] > 1)]
cltvc_df["recency"] = cltvc_df["recency"] / 7
cltvc_df["T"] = cltvc_df["T"] / 7



#############################################################
#2.BG/NBD modelinin kurulması 1-2 satırlık bir şey veri hazırlama genel anlamda uzun sürer
##############################################################
bgf = BetaGeoFitter(penalizer_coef= 0.001) #model uygulanıp parametrele bulunurken veerilecek ceza puanı
#bg-nbd en çok olabilirlik yöntemiyle beta ve gama dağılımlarının parametrelerini bulmakta ve bir tahmin yapabilmemizi sağlamak için model oluşturur.
bgf.fit(cltvc_df["frequency"],
        cltvc_df["recency"],
        cltvc_df["T"])

#############################################################
#bir hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir
###############################################################
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltvc_df["frequency"],
                                                        cltvc_df["recency"],
                                                        cltvc_df["T"]).sort_values(ascending=False).head(10)

bgf.predict(1, cltvc_df["frequency"],
               cltvc_df["recency"],
               cltvc_df["T"]).sort_values(ascending=False).head(10)
#predict bi gm-bd için geçerlidir ama gama gama için geçerli değildir.

cltvc_df["expected_purc_1_week"] = bgf.predict(1,
                                                cltvc_df["frequency"],
                                                cltvc_df["recency"],
                                                cltvc_df["T"])
#1 aylık beklenen satış geliri
bgf.predict(4,
            cltvc_df["frequency"],
            cltvc_df["recency"],
            cltvc_df["T"]).sum()
#3 aysa tüm şirketin beklenen satış sayısı nedir?
bgf.predict(4*3,
            cltvc_df["frequency"],
            cltvc_df["recency"],
            cltvc_df["T"]).sum()


#tahmin sonuçlarının değerlendirilmesi
plot_period_transactions(bgf)
plt.show()


#####################################################
#3.Gama-Gama Modelinin kurulması:
#####################################################
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltvc_df["frequency"], cltvc_df["monetary"])

ggf.conditional_expected_average_profit(cltvc_df["frequency"],
                                        cltvc_df["monetary"]).sort_values(ascending=False).head(10)

cltvc_df["expected_avarage_profit"] = ggf.conditional_expected_average_profit(cltvc_df["frequency"],
                                                                              cltvc_df["monetary"])

cltvc_df["expected_avarage_profit"].sort_values("expected_avarage_profit", ascending=False).head()#sort valuesini expected_avarage_profit e göre
#yaptım.
#################################################################
#4.bg-nbd ve gg modeli ile cltv'nin hesaplanması
#################################################################
cltv = ggf.customer_lifetime_value(bgf,
                                   cltvc_df["frequency"],
                                   cltvc_df["recency"],
                                   cltvc_df["T"],
                                   cltvc_df["monetary"],
                                   time=3, #3 aylık
                                   freq="W",#T'nin frekans bilgisi
                                   discount_rate=0.01)
cltv.head()
cltv = cltv.reset_index()#DEğişkene döndürdük.
cltv_final = cltvc_df.merge(cltv, on="Customer ID", how = "left")
cltv_final.sort_values(by="clv", ascending=False).head(10)


#bir gm-nbd: senini için düzenli olan bir müşteri recency değeri arttıkça satın alma olasılığı artar. çünkü müşteri alışveriş yaptı kısmı
#churn oluyordu. müşteri satın alam yaptıktan sonra kenara çekilir ve belli bir zaman geçer. sadece frekans sadece monetary sadece müşteri
#yaşına göre kıyaslama yapılmamalı hepsine birsen bakılıp ona göre kıyaslama yapılmalı


#################################################################
#5.Cltvye göre segmentlerin oluşturulması
##################################################################
cltv_final
cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.sort_values(by="clv", ascending=False).head(50)
cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})





#################################################################
#6.Çalışmanın fonksiyonlaştırılması
##################################################################
def create_cltv_p(dataframe, month = 3):#month son modelimiz için life time value değeri tahmini için verilen bir özellik
#1.Veri Önişleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_threshold(dataframe, "Quantity")
    replace_with_threshold(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltvc_df = df.groupby("Customer ID").agg(
        {"InvoiceDate" : [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                          lambda InvoiceDate:(today_date - InvoiceDate.min()).days],
         "Invoice": lambda Invoice: Invoice.nunique(),
         "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

    cltvc_df.columns = cltvc_df.columns.droplevel(0)
    cltvc_df.columns = ["recency", "T", "frequency", "monetary"]
    cltvc_df["monetary"] = cltvc_df["monetary"] / cltvc_df["frequency"]
    cltvc_df = cltvc_df[(cltvc_df["frequency"] > 1)]
    cltvc_df["recency"] = cltvc_df["recency"] / 7
    cltvc_df["T"] = cltvc_df["T"] / 7

#################################################
#2.BG-NBD modelinin kurulması
#################################################
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltvc_df["frequency"],
            cltvc_df["recency"],
            cltvc_df["T"])


    cltvc_df["expected_purc_1_week"] = bgf.predict(1,
                                                   cltvc_df["frequency"],
                                                   cltvc_df["recency"],
                                                   cltvc_df["T"])



    cltvc_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltvc_df["frequency"],
                                                   cltvc_df["recency"],
                                                   cltvc_df["T"])


    cltvc_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltvc_df["frequency"],
                                                   cltvc_df["recency"],
                                                   cltvc_df["T"])


    #3.Gamma-Gamma modelinin kurulması

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit((cltvc_df["frequency"], cltvc_df["monetary"])
    cltvc_df["expected_avarage_profit"] = ggf.conditional_expected_average_profit(cltvc_df["frequency"],
                                                                                  cltvc_df["monetary"]))



    #4BG-NBD ve GG modeli ile cltv nin hesaplanması
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltvc_df["frequency"],
                                       cltvc_df["recency"],
                                       cltvc_df["T"],
                                       cltvc_df["monetary"],
                                       time=month, #3aylık
                                       freq=W,
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltvc_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
    return  cltv_final
df = df_.copy()
cltv_final2 = create_cltv_p(df)

