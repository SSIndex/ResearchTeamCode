'''
File with the FeatureOptimizer class.
'''
import pandas as pd
from plotly.express import bar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class FeatureOptimizer:
  '''
    Object that handles the best features for Logistic Regression Models using a given dataset
  '''

  def __init__(self, questions_df, question_to_predict, *scores):
    X = questions_df.drop(columns=[question_to_predict])
    Y = questions_df[question_to_predict]
    self.question_to_predict = question_to_predict
    self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
      X,
      Y,
      test_size=0.2,
      random_state=42
    )
    self.scores = scores
    self.stats = pd.DataFrame(columns=['model_name', *self._x_train.columns, *map(lambda x: x.__name__, scores)])
    self.frequencies = None
  
  def add_scores(self, *scores):
    self.scores += scores
  
  def fit(self, metric, max_value=True):
    '''
    Return and set stats of the optimization process according to a given metric and if its optimal value is the maximum or minimum
    '''
    seed_question, *available_questions = self._x_train.columns
    self.optimize([ seed_question ], available_questions)
    # Filtrar las ramas finales
    self.stats = self.stats[self.stats[self.scores[0].__name__] > -1]

    sequence_metric = self.stats[metric]
    best_value = sequence_metric.max() if max_value else sequence_metric.min()
    self.stats = self.stats[self.stats[metric] == best_value]
    return self.stats
  
  def optimize(self, questions_included, questions_available):
    '''
    DP function to get all models with their performance, given questions already included and current available questions
    '''
    if len(questions_available) == 0:
      return { score.__name__ : -1 for score in self.scores }
    
    question_selected, *other_questions = questions_available
    
    # Models
    model_with_question = LogisticRegression(max_iter=1000)
    model_without_question = LogisticRegression(max_iter=1000)

    X_with_question = self._x_train[[question_selected, *questions_included]]
    X_test_with_question = self._x_test[[question_selected, *questions_included]]

    X_without_question = self._x_train[questions_included]
    X_test_without_question = self._x_test[questions_included]

    model_with_question.fit(X_with_question, self._y_train)
    model_without_question.fit(X_without_question, self._y_train)

    y_predict_with_question = model_with_question.predict(X_test_with_question)
    y_predict_without_question = model_without_question.predict(X_test_without_question)

    scores_only_X = { score.__name__ : score(y_predict_with_question, self._y_test) for score in self.scores }
    scores_no_X = { score.__name__ : score(y_predict_without_question, self._y_test) for score in self.scores }
    score_with_X = self.optimize([question_selected, *questions_included], other_questions)

    only_X_df = pd.DataFrame.from_dict({
      'model_name': [ f'{ "".join(map(lambda x: "1" if x in [ question_selected ] else "0", self._x_train.columns)) }' ],
      **{k: [ v ] for k, v in scores_only_X.items()},
      **{ q: [ 'yes' if q in [ question_selected ] else 'no' ] for q in self._x_train.columns }
    })
    best_X_df = pd.DataFrame.from_dict({
      'model_name': [ f'{ "".join(map(lambda x: "1" if x in [ question_selected, *questions_included ] else "0", self._x_train.columns)) }' ],
      **{k: [ v ] for k, v in score_with_X.items()},
      **{ q: [ 'yes' if q in [ question_selected, *questions_included ] else 'no' ] for q in self._x_train.columns }
    })
    no_X_df = pd.DataFrame.from_dict({
      'model_name': [ f'{ "".join(map(lambda x: "1" if x in questions_included else "0", self._x_train.columns)) }' ],
      **{ k: [ v ] for k, v in scores_no_X.items() },
      **{ q: [ 'yes' if q in questions_included else 'no' ] for q in self._x_train.columns }
    })

    self.stats = pd.concat([self.stats, only_X_df],    ignore_index=True)
    self.stats = pd.concat([self.stats, best_X_df],    ignore_index=True)
    self.stats = pd.concat([self.stats, no_X_df],      ignore_index=True)

    self.stats = self.stats.drop_duplicates(subset=['model_name'])

    return max([scores_only_X,  score_with_X,  scores_no_X], key=lambda x: x[self.scores[0].__name__])

  def generate_frequencies(self):
    if self.frequencies is None:
      best_models_dimensions = {}
      for dim in self._x_train.columns:
        try:
          best_models_dimensions[dim] = self.stats.value_counts(self.stats[dim])['yes']
        except KeyError:
          pass

      best_models_dimensions
      
      self.frequencies = pd.DataFrame(best_models_dimensions.items(), columns=['Dimension', 'Frecuencia'])
      self.frequencies = self.frequencies.sort_values(by=['Frecuencia'], ascending=False)

    return self.frequencies
  
  def plot_frequencies(self):
    if self.frequencies is None:
      return
    fig = bar(self.frequencies, x='Dimension', y='Frecuencia', title=f'Top Predictores de {self.question_to_predict}')

    return fig.show()

  def include_num_predictors(self, func):
    def predictors_included(*a, **kw):
      kw.update({'num_regressors': len(self._x_train.columns)})
      return func(*a, **kw)
    return predictors_included
  
  def include_num_observations(self, func):
    def observations_included(*a, **kw):
      kw.update({'num_observations': len(self._x_train.index)})
      return func(*a, **kw)
    return observations_included
