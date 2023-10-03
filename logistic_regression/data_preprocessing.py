'''
Data preprocessing file. We asume that data comes from pandas dataFrames
'''
import pandas as pd


class DataPreProcessor:
    '''
    Data preprocessing from dataframes of likert columns
    '''

    def __init__(self, data, columns_conversion=None):
        self.data = data
        self.binarized = False
        self.transformed = False
        if columns_conversion is not None:
            self.additive_combine(columns_conversion)
        self.binarize_data()

    def binarize_data(self):
        '''
        Convert data from likert scale (1..5) to binary (0, 1)
        '''
        def base_sentiment(value):  # 1, 2 -> 0 (neg); 3 -> 1 (neu); 4, 5 -> 2 (pos)
            if value > 3:
                return 2
            elif value < 3:
                return 0
            return 1
        def aproximado_neg(value):
            if value in (0, 1):
                return 0
            return 1
        def aproximado_pos(value):
            if value in (1, 2):
                return 0
            return 1

        if not self.binarized:
            data_base_sentiment = pd.DataFrame()
            for col in self.data:
                data_base_sentiment[col] = self.data[col].apply(base_sentiment)

            data_aproximada_neg = pd.DataFrame()
            data_aproximada_pos = pd.DataFrame()

            for col in data_base_sentiment:
                data_aproximada_neg[col] = data_base_sentiment[col].apply(aproximado_neg)
                data_aproximada_pos[col] = data_base_sentiment[col].apply(aproximado_pos)

            self.data = pd.concat(
                [data_aproximada_pos, data_aproximada_neg],
                ignore_index=True
            )
        return self.data

    def additive_combine(self, column_transform):
        '''
        Group columns and create new data from colum_transform
        the premise is that the range of linkert are in range 1..5
        the combined not normalized are from K..5*K where K is the number of columns
        that belongs to the group new column.
        column_transform is a dataframe with columns named "Pregunta", "Dimension"
        '''
        if not self.transformed:
            group_counts = { val: count for val, count in column_transform['Dimension'].value_counts().iteritems() }
            additive_combine_df = pd.DataFrame()
            for group in group_counts.keys():
                cols_in_group = column_transform[column_transform['Dimension'] == group]['Pregunta'].unique()
                additive_combine_df[group] = self.data[cols_in_group].sum(axis=1)
            self.data = additive_combine_df
        return self.data

