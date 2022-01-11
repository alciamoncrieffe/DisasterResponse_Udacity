import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load messages and categories from csv files and return merged data as dataframe
    Args 
        string : messages_filepath - csv path to messages file
        string : categories_filepath - csv path to categories file
    Return
        dataframe : df - merged data
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on='id')
    return df


def clean_data(df):
    '''
    split categories into separate columns and remove duplicates
    Args 
        dataframe : df - loaded merged message and categories data
    Return
        dataframe : df - cleaned data
    '''
    #split categories into separate columns
    categories = df['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe to use as column headers
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df = df.drop(columns=['categories'] )
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    save data to sql database
    Args 
        dataframe : df
        string : database_filepath - csv path to messages sql database
    Return
        none
    '''
    engine = create_engine(database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()