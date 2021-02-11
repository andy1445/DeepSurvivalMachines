# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Utility functions to load standard datasets to train and evaluate the
Deep Survival Machines models.
"""


import io
import pkgutil

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torchvision

def increase_censoring(e, t, p):
  if p==0.:
    return e, t
  uncens = np.where(e == 1)[0]
  mask = np.random.choice([False, True], len(uncens), p=[1-p, p])
  toswitch = uncens[mask]

  e[toswitch] = 0
  t_ = t[toswitch]

  newt = []
  for t__ in t_:
    newt.append(np.random.uniform(1, t__))
  t[toswitch] = newt

  return e, t

def _load_framingham_dataset(sequential):
  """Helper function to load and preprocess the Framingham dataset.

  The Framingham Dataset is a subset of 4,434 participants of the well known,
  ongoing Framingham Heart study [1] for studying epidemiology for
  hypertensive and arteriosclerotic cardiovascular disease. It is a popular
  dataset for longitudinal survival analysis with time dependent covariates.

  Parameters
  ----------
  sequential: bool
    If True returns a list of np.arrays for each individual.
    else, returns collapsed results for each time step. To train
    recurrent neural models you would typically use True.

  References
  ----------
  [1] Dawber, Thomas R., Gilcin F. Meadors, and Felix E. Moore Jr.
  "Epidemiological approaches to heart disease: the Framingham Study."
  American Journal of Public Health and the Nations Health 41.3 (1951).

  """

  data = pkgutil.get_data(__name__, 'datasets/framingham.csv')
  data = pd.read_csv(io.BytesIO(data))

  dat_cat = data[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS',
                  'educ', 'PREVCHD', 'PREVAP', 'PREVMI',
                  'PREVSTRK', 'PREVHYP']]
  dat_num = data[['TOTCHOL', 'AGE', 'SYSBP', 'DIABP',
                  'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']]

  x1 = pd.get_dummies(dat_cat).values
  x2 = dat_num.values
  x = np.hstack([x1, x2])

  time = (data['TIMEDTH'] - data['TIME']).values
  event = data['DEATH'].values

  x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
  x_ = StandardScaler().fit_transform(x)

  if not sequential:
    return x_, time, event
  else:
    x, t, e = [], [], []
    for id_ in sorted(list(set(data['RANDID']))):
      x.append(x_[data['RANDID'] == id_])
      t.append(time[data['RANDID'] == id_])
      e.append(event[data['RANDID'] == id_])
    return x, t, e

def _load_pbc_dataset(sequential):
  """Helper function to load and preprocess the PBC dataset

  The Primary biliary cirrhosis (PBC) Dataset [1] is well known
  dataset for evaluating survival analysis models with time
  dependent covariates.

  Parameters
  ----------
  sequential: bool
    If True returns a list of np.arrays for each individual.
    else, returns collapsed results for each time step. To train
    recurrent neural models you would typically use True.

  References
  ----------
  [1] Fleming, Thomas R., and David P. Harrington. Counting processes and
  survival analysis. Vol. 169. John Wiley & Sons, 2011.

  """

  data = pkgutil.get_data(__name__, 'datasets/pbc2.csv')
  data = pd.read_csv(io.BytesIO(data))

  data['histologic'] = data['histologic'].astype(str)
  dat_cat = data[['drug', 'sex', 'ascites', 'hepatomegaly',
                  'spiders', 'edema', 'histologic']]
  dat_num = data[['serBilir', 'serChol', 'albumin', 'alkaline',
                  'SGOT', 'platelets', 'prothrombin']]
  age = data['age'] + data['years']

  x1 = pd.get_dummies(dat_cat).values
  x2 = dat_num.values
  x3 = age.values.reshape(-1, 1)
  x = np.hstack([x1, x2, x3])

  time = (data['years'] - data['year']).values
  event = data['status2'].values

  x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
  x_ = StandardScaler().fit_transform(x)

  if not sequential:
    return x_, time, event
  else:
    x, t, e = [], [], []
    for id_ in sorted(list(set(data['id']))):
      x.append(x_[data['id'] == id_])
      t.append(time[data['id'] == id_])
      e.append(event[data['id'] == id_])
    return x, t, e

def _load_support_dataset():
  """Helper function to load and preprocess the SUPPORT dataset.

  The SUPPORT Dataset comes from the Vanderbilt University study
  to estimate survival for seriously ill hospitalized adults [1].

  Please refer to http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc.
  for the original datasource.

  References
  ----------
  [1]: Knaus WA, Harrell FE, Lynn J et al. (1995): The SUPPORT prognostic
  model: Objective estimates of survival for seriously ill hospitalized
  adults. Annals of Internal Medicine 122:191-203.

  """

  data = pkgutil.get_data(__name__, 'datasets/support2.csv')
  data = pd.read_csv(io.BytesIO(data))
  x1 = data[['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp',
             'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun',
             'urine', 'adlp', 'adls']]

  catfeats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
  x2 = pd.get_dummies(data[catfeats])

  x = np.concatenate([x1, x2], axis=1)
  t = data['d.time'].values
  e = data['death'].values

  x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
  x = StandardScaler().fit_transform(x)

  remove = ~np.isnan(t)
  return x[remove], t[remove], e[remove]

def _load_mnist(load_test=False, censoring_percent=.5):
  """Helper function to load and preprocess the MNIST dataset.

  The MNIST database of handwritten digits, available from this page, has a
  training set of 60,000 examples, and a test set of 10,000 examples.
  It is a good database for people who want to try learning techniques and
  pattern recognition methods on real-world data while spending minimal
  efforts on preprocessing and formatting [1].

  Please refer to http://yann.lecun.com/exdb/mnist/.
  for the original datasource.

  References
  ----------
  [1]: LeCun, Y. (1998). The MNIST database of handwritten digits.
  http://yann.lecun.com/exdb/mnist/.

  """
  np.random.seed(1234)
  transform = torchvision.transforms.Compose(
      [torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize((0.1307,), (0.3081,))]
  )
  mnist_train = torchvision.datasets.MNIST(root='datasets/',
                                           train=True,
                                           download=True,
                                           transform=transform)
    
  mnist_test = torchvision.datasets.MNIST(root='datasets/',
                                           train=False,
                                           download=True,
                                           transform=transform)
    
  t_train = mnist_train.targets.numpy()+1
  t_test = mnist_test.targets.numpy()+1

  x_train = mnist_train.data.numpy().astype(float)
  x_train = np.expand_dims(x_train, 1).astype(float)
  x_train /= 255.0

  x_test = mnist_test.data.numpy().astype(float)
  x_test = np.expand_dims(x_test, 1).astype(float)
  x_test /= 255.0

  t_train = np.random.lognormal(t_train)
  t_test = np.random.lognormal(t_test)

  e_train, t_train = increase_censoring(np.ones(len(t_train)), t_train, censoring_percent)
  e_test, t_test = increase_censoring(np.ones(len(t_test)), t_test, censoring_percent)

  if load_test == False:
    return x_train, t_train, e_train
  else:
    return x_train, t_train, e_train, x_test, t_test, e_test

def load_dataset(dataset='SUPPORT', **kwargs):
  """Helper function to load datasets to test Survival Analysis models.

  Currently implemented datasets include:

  **SUPPORT**: This dataset comes from the Vanderbilt University study
  to estimate survival for seriously ill hospitalized adults [1].
  (Refer to http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc.
  for the original datasource.)

  **PBC**: The Primary biliary cirrhosis dataset [2] is well known
  dataset for evaluating survival analysis models with time
  dependent covariates.

  **FRAMINGHAM**: This dataset is a subset of 4,434 participants of the well
  known, ongoing Framingham Heart study [3] for studying epidemiology for
  hypertensive and arteriosclerotic cardiovascular disease. It is a popular
  dataset for longitudinal survival analysis with time dependent covariates.

  References
  -----------

  [1]: Knaus WA, Harrell FE, Lynn J et al. (1995): The SUPPORT prognostic
  model: Objective estimates of survival for seriously ill hospitalized
  adults. Annals of Internal Medicine 122:191-203.

  [2] Fleming, Thomas R., and David P. Harrington. Counting processes and
  survival analysis. Vol. 169. John Wiley & Sons, 2011.

  [3] Dawber, Thomas R., Gilcin F. Meadors, and Felix E. Moore Jr.
  "Epidemiological approaches to heart disease: the Framingham Study."
  American Journal of Public Health and the Nations Health 41.3 (1951).

  Parameters
  ----------
  dataset: str
      The choice of dataset to load. Currently implemented is 'SUPPORT',
      'PBC' and 'FRAMINGHAM'.
  **kwargs: dict
      Dataset specific keyword arguments.

  Returns
  ----------
  tuple: (np.ndarray, np.ndarray, np.ndarray)
      A tuple of the form of (x, t, e) where x, t, e are the input covariates,
      event times and the censoring indicators respectively.

  """
  sequential = kwargs.get('sequential', False)
  load_test = kwargs.get('load_test', False)
  censoring_percent = kwargs.get('censoring_percent', False)

  if dataset == 'SUPPORT':
    return _load_support_dataset()
  if dataset == 'PBC':
    return _load_pbc_dataset(sequential)
  if dataset == 'FRAMINGHAM':
    return _load_framingham_dataset(sequential)
  if dataset == 'MNIST':
    return _load_mnist(load_test, censoring_percent)
  else:
    raise NotImplementedError('Dataset '+dataset+' not implemented.')
