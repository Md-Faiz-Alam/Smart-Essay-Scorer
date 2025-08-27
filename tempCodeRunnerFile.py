import nltk
from nltk.tokenize import sent_tokenize

text = "This is a test. Let's see if it works."
sentences = sent_tokenize(text)
print(sentences)