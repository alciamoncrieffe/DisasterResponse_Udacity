import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
#from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    load messages and categories from sql database and return data as 
    X features, Y output and Y column names
    Args 
        string : database_filepath - csv path to messages sql database
    Return
        dataframe : X - messages
        dataframe : Y - category columns
        list : category_names
    '''
    engine = create_engine(database_filepath)
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    Y = df.drop(columns=['message','id','original','genre'])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    clean text, remove punctuation, lemmatize, tokenize and remove stopwords
    Args 
        string : text - message data
    Return
        list : clean_tokens
    '''
    text = re.sub(r"[^a-zA-Z0-9]"," ", text) 
    #tokens = ne_chunk(pos_tag(word_tokenize(text))) could use pos-tag at this point?
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() #lemmatize, lowercase and remove spaces
        if clean_tok not in stopwords.words("english"): 
            clean_tokens.append(clean_tok) 
    return clean_tokens


def build_model():
    '''
    define model pipeline and parameters, and return gridsearch model
    Args 
        none
    Return
        model : cv
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    parameters = {
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__ngram_range': ((1, 1),(1,2)),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10,50,100],
        'clf__estimator__min_samples_split': [2,3,4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    use model to create predictions and compare to Y_test
    Args 
        model : model
        df : X_test
        df : Y_test
        list : category_names
    Return
        none
    '''
    Y_pred = model.predict(X_test)
    for x in range(len(category_names)):
        print(category_names[x].upper())
        print(classification_report(np.array(Y_test)[:,x], Y_pred[:,x]))


def save_model(model, model_filepath):
    '''
    save model to pickle model_filepath
    Args 
        model : model
        string : model_filepath - csv path to save pickle model
    Return
        none
    '''
    pickle.dump(cv, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()