# Disaster Response Pipeline Project

## Table of Contents

1. [Installation](#Install)
2. [Description](#Description)
3. [Project Components](#Project)
4. [Instruction](#Instruction)
5. [Acknowledgement](#acknowledgement)


<a name = "Install"></a>
### Installation:
This project used HTML, bootstrap and python, and requires the following Python packages: pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys.

<a name = "Description"></a>
### Description:
This project is the part of Udacity nanodegree programm.In the Project, you'll find a data set containing real messages that were sent during disaster events. We are creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. The dataset provided by figure8.

<a name = "Project"></a>
### Project Components:
Three key component of the project

1. ETL Pipeline
In a Python script, process_data.py
* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

2. ML Pipeline
In a Python Script, train_classifier.py
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

3. Flask Web app
* Web app to show models result in real time

<a name = "Instruction"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. After this open another terminal and type env|grep WORK this will give you the spaceid it will start with view*** and some characters    after that
5. Now open your browser window and type `http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN`  replace the whole workspaceid with your space id and workspace domain to `udacity-student-workspaces.com` you got in the step 4
6. Press enter and the app should now run for you
    `http://view6914b2f4-3001.udacity-student-workspaces.com` `http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` 
    
<a name = "acknowledgement"></a>
### Acknowledgements
* [Udacity](https://www.udacity.com/) for this nanodegree assignment
* [Figure Eight](https://www.figure-eight.com/) for providing the datasets for this project
