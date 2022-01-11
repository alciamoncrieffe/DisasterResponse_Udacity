# Disaster Response Pipeline Project

### Description / Motivation

To build a model which analyzes disaster messages and uses machine learning to classify new messages as part of an API.
This is part of a Udacity data science project intended to demonstrate ETL and ML pipeline production.

## Installation
### Dependencies:
 Use  `pip` to install required packages.
`pip install -r requirements.txt`

### Folder Breakdown: 
- data
    - `disaster_messages.csv` - `message`, message `id`, `genre` (source), and `original` columns. The `message` column should contain messages in english which were received by disaster response units. The untranslated version should be available in the `original` column.
    - `disaster_categories.csv` - Columns of 36 categories which the message `id` column has already been aligned to
    - `process_data.py` - Contains the functions required to load in the message and category csv files, merge, clean the data and save the merged data to a sql database.
- model
    - `train_classifier.py` - Contains functions to load the processed message data and use to train a multi-output classifier, which could then be used to predict which of the 36 categories new messages would apply to.
- app
    - `run.py` - Providing the 2 previous python file have been run, producing a sql database and trained model, running this code will load web app backend and using these objects for visualisations and to create predictions to the user.
    - templates - Web app html files
        - `go.html` 
        - `master.html`

## Instructions
1. Run the following commands in the project's root directory to set up your process the raw message and category data, producing a database of cleaned data, and a subsequent trained model.

    - To run *ETL pipeline* that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run *ML pipeline* that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to view the web app, which should display the navbar, message classifier, and graphs derived from the training data.

## Contributing

Feel free to create new branches or contribute. Next steps would be to add resource links to the webpage model predictions for the user to access for further aid. 

## Licensing

See package license.txt file.
