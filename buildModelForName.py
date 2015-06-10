import pickle
from customTransformers import CleaningContextTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import dill

name = 'mary'

context = pickle.load(open('processed/context_' + name + '.pkl', 'rb'))
link = pickle.load(open('processed/link_' + name + '.pkl', 'rb'))

model = Pipeline([('cleaning', CleaningContextTransformer()), ('vectorize', DictVectorizer()), ('decision tree', DecisionTreeClassifier(max_depth = 500))])

model = model.fit(context, link)

dill.dump(model, open('models/model_' + name + '.dill', 'wb'))
