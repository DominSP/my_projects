getwd()
Suicide_Detection <- read.csv("Suicide_Detection.csv", sep = ",") #dane należy podbrać z Kaggle https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?fbclid=IwAR0s9EECEHd7B2xYubTytFXYVxoW4thbEqESogFy_K5iTwL6o1v48WwYzPY
pomocnicze_dane<-Suicide_Detection[1:15000,]
pomocnicze_dane[40,2]
head(Suicide_Detection)
summary(pomocnicze_dane)
str(pomocnicze_dane)

# X- jakies liczby (int)
# teskt (charakter)
# class- przypisuje non suicide, albo suicide (charakter)

str(pomocnicze_dane[,c(2,3)])

Data<-pomocnicze_dane[,c(2,3)] #pozbycie sie pierwszej kolumny
summary(Data)
str(Data)

#konwerujemy class tak aby przydzielic odpowiednie liczby slowom suicide, non-suicide
Data$class<-factor(Data$class)

Data$class[1:10]
str(Data$class)
summary(Data$class)

library(tm)
#utworzenie korpusu danych, jako zbiór tekstów
Data_corpus<-VCorpus(VectorSource(Data$text))
print(Data_corpus)

#wyświetlmy tekst danej wiadmości
as.character(Data_corpus[[344]])

#usuwamy niepotrzebne biale znaki
Data_corpus_clean<-tm_map(Data_corpus, stripWhitespace)
as.character(Data_corpus_clean[[23]])

#robimy porządek z literami
#duze litery na male litery, str 33 prez 3
Data_corpus_clean<-tm_map(Data_corpus_clean,content_transformer(tolower))
as.character(Data_corpus_clean[[8]])

#usuniemy slowa wypelniajace takie jak 'albo' 'lub', nie niosa zadnych informacji
stopwords()
Data_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwords())
as.character(Data_corpus_clean[[344]])

#redukujemy slowa do ich rdzenia
#Data_corpus_clean<-tm_map(Data_corpus_clean,stemDocument)
#robi błędy, urywa ostatnie litery

#usuwamy znaki interpunkcyjne
Data_corpus_clean<-tm_map(Data_corpus_clean,removePunctuation)
as.character(Data_corpus_clean[[267]])

#usuwanie liczb
Data_corpus_clean<-tm_map(Data_corpus_clean,removeNumbers)
as.character(Data_corpus_clean[[8]])

Data_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwords())

#usuwamy niepotrzebne biale znaki
Data_corpus_clean<-tm_map(Data_corpus_clean, stripWhitespace)
as.character(Data_corpus_clean[[17]])

stopwordsextra9 <- c("them","done","more","ing","ll","never","idk","past","du","everything","whenever","say","used","op","like","gets")
stopwordsextra8 <- c("dont","just","now","one","due","yet"," ive","ive ","get","why","hes","all","when","isnt","still","con")
stopwordsextra7 <- c("really","during","their","own","because","what","was","ever","isn't","had","up","down","once","from","she","he","i","you","told","without")
stopwordsextra6 <- c("want","if","going","go","goes","when","doesn't","here","don't"," 'll","'ll","emojis","new","old","bot","whether","memes","let","gather")
stopwordsextra5 <- c("won't","by","then","but","will","a","said","would","have","least","thus","everyone","mani","bit","every","always")
stopwordsextra4 <- c("one","with","at","very","off","how","on","do","am","be","has","you","could"," 's","else","there","receny","s","u","i")
stopwordsextra3 <- c("it","of","for","to","don't","much","been","so","in","not","or","any","can","now","can't","many","much")
stopwordsextra2 <- c("ive","cant","just","got","this","also","even","some","hes","theres","since","shes","wont","its","thats","that","the","anyone","dont","and")
stopwordsextra <- c("i'm","she's","is","are","im","my","mine","her","his","i","me","my","myself","we","our","ours","can't","didn't","didnt","i've")


suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra)
suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra2)
suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra3)
suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra4)
suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra5)
suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra6)
suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra7)
suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra8)
suicide_corpus_clean<-tm_map(Data_corpus_clean,removeWords,stopwordsextra9)

#tworzymy macierz dla naszego korpusu ktora przechowuje ilosc slow
macierz_dtm<-DocumentTermMatrix(Data_corpus_clean)

#tworzymy zbiór uczący i testowy
dl<-length(Data[,1])
tr<-round(0.7*dl,0)
te<-dl-tr
trte<-tr+1
zbioruczacy_train<-macierz_dtm[1:tr,]
zbiortestowy_test<-macierz_dtm[trte:dl,]

#zapisujemy etykiety dla zbioru ucz i dla test
dane_train_labels<-Data[1:tr,]$class
dane_test_lables<-Data[trte:dl,]$class

#sprawdzamy czy odsetek suicide jest podobny w zb ucz jak i test
prop.table(table(dane_train_labels))
prop.table(table(dane_test_lables))

#dzielimy dane na suicide oraz non-suicide
suicide<-subset(Data, class=="suicide")
non_suicide<-subset(Data, class=="non-suicide")

stopwords()

library(wordcloud)
#chmurki sie troche robia, duzo danych
wordcloud(suicide$text, max.words=40, scale=c(2.5,2))
wordcloud(non_suicide$text, max.words=60, scale=c(2,2))

#zapisujemy słowa, które występują co najmniej 15 razy w zb uczacym
dane_freq_words<-findFreqTerms(zbioruczacy_train,15)
str(dane_freq_words)
dane_freq_words

#tworzymy macierz DTM
dane_dtm_freq_train<-zbioruczacy_train[,dane_freq_words]
dane_dtm_freq_test<-zbiortestowy_test[,dane_freq_words]

#funcja ktora konwertuje liczbe wyrazu w wiadmosci na yes jesli wystepuje no jesli nie

convert_counts<-function(x) {
  x<-ifelse(x>0,"Yes","No")
  return(x)
}

#stosujemy powyzsza funkcje do kolumn
#tu juz za dlugo dziala
nowe_dane_train<-apply(dane_dtm_freq_train, MARGIN = 2, convert_counts)
nowe_dane_test<-apply(dane_dtm_freq_test, MARGIN = 2, convert_counts)

#budujemy model do przewidywania typu wiadomosci
library(e1071)
dane_classifier<-naiveBayes(nowe_dane_train,dane_train_labels)

dane_test_pred<-predict(dane_classifier, nowe_dane_test)


library(gmodels)
CrossTable(dane_test_pred, dane_test_lables,
           prop.chisq=FALSE, prop.c=FALSE, prop.r=FALSE,
           dnn=c('predicted','actual'))

#spróbujmy z wygładzeniem Laplace'a

dane_classifier2<-naiveBayes(nowe_dane_train, dane_train_labels, laplace=1)

dane_test_pred2<-predict(dane_classifier2, nowe_dane_test)

CrossTable(dane_test_pred2, dane_test_lables,
           prop.chisq = FALSE, prop.c=FALSE, prop.r=FALSE,
           dnn=c('predicted','actual'))
