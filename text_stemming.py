from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

#plain usage of the CountVectorizer
vocab = ['The swimmer likes swimming so he swims.']
vec = CountVectorizer().fit(vocab)

sentence1 = vec.transform(['The swimmer likes swimming.'])
sentence2 = vec.transform(['The swimmer swims.'])

#Nome está diferente da pergunta do stackoverflow pois o nome do método foi alterado em versões mais recentes do sklearn
print('Vocabulary: %s' %vec.get_feature_names_out())
print('Sentence 1: %s' %sentence1.toarray())
print('Sentence 2: %s' %sentence2.toarray())


#Now, let's say I want to remove stop words and stem the words. One option would be to do it like so:
#######
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
########

vect = CountVectorizer(tokenizer=tokenize, stop_words='english', token_pattern=None)

vect.fit(vocab)

sentence1 = vect.transform(['The swimmer likes swimming.'])
sentence2 = vect.transform(['The swimmer swims.'])

print('Vocabulary: %s' %vect.get_feature_names_out())
print('Sentence 1: %s' %sentence1.toarray())
print('Sentence 2: %s' %sentence2.toarray())