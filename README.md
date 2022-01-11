# Disaster Response Pipeline Project

### Description / Motivation

To build a model which analyzes disaster messages and uses machine learning to classify new messages as part of an API.
This is part of a Udacity data science project intended to demonstrate ETL and ML pipeline production.

## Installation
 Use  `pip` to install required packages.
`pip install -r requirements.txt`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to view the web app, which should display the navbar, message classifier, and graphs derived from the training data.

## Contributing

Feel free to create new branches or contribute. 
Next steps would be to add links to the webpage model predictions for the user to access for further aid. 

## Licensing

See package license.txt file.
