""" 
Pickle and save any data type to an external file. 


Please cite the following paper when using this code:
    Stefano Giglio, Bryan Kelly, Serhiy Kozak "Equity Term Structures without Dividend Strips Data"
    Journal of Finance, 2024. Forthcoming

====================
Author: Serhiy Kozak
Date: November 2023
"""

import pickle, bz2
import tensorflow as tf
import numpy as np

def save_data(data, filename, path=None):
  if path is None:
    path = ''
    
  # unpack any Tensors
  if isinstance(data, list):
    d = []
    for i, e in enumerate(data):
      d.append(e.numpy().tolist() if tf.is_tensor(e) else e)
  elif isinstance(data, dict):
    d = {}
    for k, v in data.items():
      d[k] = v.numpy().tolist() if tf.is_tensor(v) else v
  elif tf.is_tensor(data):
    d = data.numpy().tolist()
  elif isinstance(data, np.ndarray):
    d = data.tolist()
  else:
    d = data
      
  # with bz2.BZ2File(respath + filename + '.pickle.bz2', 'w') as f: 
  with open(path + filename + '.pickle', 'wb') as f:
    pickle.dump(d, f)
