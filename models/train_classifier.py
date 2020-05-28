import sys

import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(database_filepath):
    """ Load the database 
    Fucntion to load the database from the given filepath and process them as X, Y and category_names.
    

    Args : 
        Databased filepath

    Returns: 
        Returns the Features X & target y along with target columns names catgeory_names
    """
    databasePath = 'sqlite:///' + database_filepath
    engine = create_engine(databasePath)
    df = pd.read_sql_table("Disasters", con = engine)
    print(df.head(3))
    X = df['message']
    Y = df[df.columns[5:]]
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    """ Used to tokenize
    Fucntion help to tokenize the text messages 

    Args : 
        Text messages

    Returns: 
        A list of clean tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Building model with pipeline
    Function to build a model, create pipeline, use of gridsearchcv

    Args: 
        N/A

    Return: 
        Returns the model

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters =  {
             'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 

    cv2 = GridSearchCV(pipeline, param_grid=parameters)
    return cv2
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluating model accuracy
    Function to evaluate a model and return the classification report and accurancy score.
    
    Args: 
        Model, X_test, y_test, Catgegory_names
    
    Retruns: 
        Prints the Classification report & Accuracy Score
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))
    pass


def save_model(model, model_filepath):
    """ Save model in a pickle file
    This Function helps to save the model ub pickle file

    Args: 
        model and file path to save model

    Returns: 
        save the model as pickle file in the give filepath 

    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    """
        This function runs the main function which call all other function
            Load_data()
            tokenize()
            build_model()
            evaluate_model()
            save_model()
    """
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