""" 
Utility functions. 


Please cite the following paper when using this code:
    Stefano Giglio, Bryan Kelly, Serhiy Kozak "Equity Term Structures without Dividend Strips Data"
    Journal of Finance, Forthcoming

====================
Author: Serhiy Kozak
Date: November 2023
"""


import pandas as pd
import numpy as np
import json, yaml


#| ### Tables helper functions

def output_table(tbl, filename=''):
  tbl.to_latex(float_format=lambda x: '%10.4f' % x)

def summary_table(data, filename='', transpose=False):
  summ = data.describe()
  if transpose:
    summ = summ.T
  print(summ)
  output_table(summ, filename)


# latex tables
def writefilestr(filename, text):
  with open(filename, 'w') as f:
    f.write(text)

def latextable(filename, df, header=False, float_format='%.1f', na_rep='-'):
  writefilestr(filename, df.to_csv(header=header, lineterminator='\\\\ \n', quotechar=' ',
                                   float_format=float_format, sep='&', na_rep=na_rep))
# index=['R', 'D/P'], columns=labels[:npc]).to_latex(col_space=10, float_format=lambda x: '%.1f' % x))



#| ### Helper functions for printing

# helpers for log printing
def log_line(line, f=None):
  tf.print(line)
  if f != None:
      f.write(line + '\n')

def vec2str(vec, fmt='2.0f', sep='|'):
  return sep.join([('{:'+fmt+'}').format(v) for v in vec])


# Load JSON config file
def load_json_config(sn):
    with open('config/{}.json'.format(sn), 'r') as file:
        config = json.load(file)
    return config 



def load_yaml_config(sn):
    ## define custom tag handler
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])
    
    ## register the tag handler
    yaml.add_constructor('!join', join)


    with open('config/{}.yaml'.format(sn), 'r') as file:
      try:
          return yaml.load(file, Loader=yaml.FullLoader)
      except yaml.YAMLError as exc:
          print(exc)
    return None
