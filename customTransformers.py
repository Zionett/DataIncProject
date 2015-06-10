from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

class SelectNameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nameToBeSelected):
        self.name = nameToBeSelected.lower()
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):  # X is a list of tuples with the format (anchor, context, wikilink)
        nameMention = []
        for mention in X:
            firstName = mention[0].split()[0]
            if firstName.lower() == self.name:
                nameMention.append((mention[1], mention[2]))
        return nameMention
    
def removeNonAscii(stringIn):
    newString = ''
    for c in stringIn:
        if c in (string.ascii_lowercase + ' ' + string.digits):
            newString += c
    return newString

def removeStopwords(words):
    stop = stopwords.words('english')
    stop += [u'us', u'may']
    newList = []
    for word in words:
        if word not in stop:
            newList.append(word)
    return newList

def removeNumbers(words):
    newList = []
    for word in words:
        flag = 0
        for c in string.digits:
            if c in word:
                flag = 1
                break
        if flag == 0:
            newList.append(word)
    return newList 
    
class CleaningContextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.s = SnowballStemmer('english')
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):      # X is a list of strings containing the context
        df = pd.DataFrame(X)
        df.columns = ['context']
        df.context = df.context.apply(lambda x: str.decode(x, 'utf8'))
        df.context = df.context.apply(lambda sent: string.replace(sent, '. ', ' '))
        df.context = df.context.apply(lambda sent: string.replace(sent, ', ', ' '))
        df.context = df.context.apply(lambda sent: string.replace(sent, '-', ' '))
        df.context = df.context.apply(string.lower)
        df.context = df.context.apply(removeNonAscii)
        df.context = df.context.apply(string.split)
        df.context = df.context.apply(removeStopwords)
        df.context = df.context.apply(removeNumbers)
        df.context = df.context.apply(lambda words: [self.s.stem(word) for word in words if len(word) > 2])
        df.context = df.context.apply(lambda words: dict([(word, 1) for word in words]))
        return df.context