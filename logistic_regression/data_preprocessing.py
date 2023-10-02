'''
Data preprocessing file. We asume that data comes from pandas dataFrames
'''

class DataPreProcessor:
    '''
    Data preprocessing from dataframes of likert columns
    '''

    def __init__(self, data, columns_conversion=None):
        self.data = data
        if columns_conversion is not None:
            self.additive_combine(columns_conversion)
        self.binarize_data()

    def binarize_data(self):
        '''
        Convert data from likert scale (1..5) to binary (0, 1)
        '''

    def additive_combine(self, column_transform):
        '''
        Group columns and create new data from colum_transform
        '''
