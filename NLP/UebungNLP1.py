from nltk import tokenize

satz = "Hallo."
satz = tokenize.sent_tokenize(satz)
print(satz)