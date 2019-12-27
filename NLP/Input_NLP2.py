import nltk, re, string, collections
from nltk.util import ngrams # function for making ngrams

text = "There is a nice restaurant in the center of New York. However, in the north of New York food is more cheap. It is more suited for visitors in their mid twenties, as it is a good alternative."

# XML markup entfernen
text = re.sub('<.*>','',text)

# "ENDOFARTICLE." entfernen
text = re.sub('ENDOFARTICLE.','',text)

# Satzzeichen entfernen ausser !
text_ohne_satzzeichen = "[" + re.sub("\.","",string.punctuation) + "]"
text = re.sub(text_ohne_satzzeichen, "", text)

tokenized = text.split()

listBigrams = nltk.bigrams(tokenized)

freq_bi = nltk.FreqDist(listBigrams)

fdist = nltk.FreqDist(freq_bi)
for k,v in fdist.items():
    print (k,v)