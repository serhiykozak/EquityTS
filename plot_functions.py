""" 
Plotting functions. 


Please cite the following paper when using this code:
    Stefano Giglio, Bryan Kelly, Serhiy Kozak "Equity Term Structures without Dividend Strips Data"
    Journal of Finance, 2024. Forthcoming

====================
Author: Serhiy Kozak
Date: November 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import savedata as sv
#import seaborn as sns

## uncomment for LaTeX plots (slower)
#from matplotlib import rc
##rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
#rc('font', size=14)
#rc('legend', fontsize=13)
##rc('text.latex', preamble=r'\usepackage{cmbright}')

axfont = {'fontsize':12} #, 'fontname': 'Computer Modern Roman'} #Comic Sans MS, Times New Roman
## matplotlib.get_cachedir()

fig_size_quarter_wide = (8.5, 11/4) # inches, full width, quarter height 


def maturity_plot(x, se=None, filename='', title='', alpha=.2, se_band=2., show=True, newfig=True, line_color=None, legends=None):
  if newfig:
    # You typically want your plot to be ~1.33x wider than tall.  
    # Common sizes: (10, 7.5) and (12, 9)  
    plt.figure(figsize=fig_size_quarter_wide)  
  
  # if line_color is None:
  #   line_color = "#3F5D7D"
    
  # Remove the plot frame lines. They are unnecessary chartjunk.  
  ax = plt.gca() #subplot(111)  
  ax.spines["top"].set_visible(False)  
  ax.spines["right"].set_visible(False)  
    
  # Ensure that the axis ticks only show up on the bottom and left of the plot.  
  # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
  ax.get_xaxis().tick_bottom()  
  ax.get_yaxis().tick_left()  
    
  # Limit the range of the plot to only where the data is.  
  # Avoid unnecessary whitespace.  
  maturities = np.array(range(x.shape[0])) + 1
  plt.xlim(maturities[0], maturities[-1])  

  # title (y label)
  plt.ylabel(title, **axfont)  
  #plt.title("Title", fontsize=22)  
  plt.xlabel('Maturity (years)', **axfont)  

  # plot    
  plt.plot(maturities, x, color=line_color, lw=2)  
  plt.grid(True)

  if legends is not None:
    # plt.legend(legends, frameon=False, loc=(1.04, 0))#, ncol=len(legends))
    plt.legend(legends, frameon=False, loc='upper center', ncol=len(legends))#, loc=(1.04, 0))#, ncol=len(legends))
    
  # Use matplotlib's fill_between() call to create error bars.  
  # Use the dark blue "#3F5D7D" as a nice fill color.  
  if se is not None:
    plt.fill_between(maturities, (x - se_band*se),  (x + se_band*se), alpha=alpha, color=line_color, label='_nolegend_')  
    
  # Finally, save the figure as a PDF, and show in log.  
  if len(filename):
    plt.savefig(filename + '.pdf', bbox_inches="tight")  
    print(filename + '.pdf')
  if show:
    # plt.show()  
    sv.save_data([x, se, filename, title], filename)




def maturity_plot2(x, se=None, se2=None, filename='', title='', alpha=.2, se_band=2., show=True, newfig=True, line_color=None, line_color2=None, line_style=None, legends=None, hatch=None):
  if newfig:
    # You typically want your plot to be ~1.33x wider than tall.  
    # Common sizes: (10, 7.5) and (12, 9)  
    plt.figure(figsize=fig_size_quarter_wide)  

  if line_color2 is None:
    line_color2 = line_color
    
  # if line_style is None:
  #   line_style = "-"
    
  # Remove the plot frame lines. They are unnecessary chartjunk.  
  ax = plt.gca() #subplot(111)  
  ax.spines["top"].set_visible(False)  
  ax.spines["right"].set_visible(False)  
    
  # Ensure that the axis ticks only show up on the bottom and left of the plot.  
  # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
  ax.get_xaxis().tick_bottom()  
  ax.get_yaxis().tick_left()  
    
  # Limit the range of the plot to only where the data is.  
  # Avoid unnecessary whitespace.  
  maturities = np.array(range(x.shape[0])) + 1
  plt.xlim(maturities[0], maturities[-1])  

  # title (y label)
  plt.ylabel(title, **axfont)  
  #plt.title("Title", fontsize=22)  
  plt.xlabel('Maturity (years)', **axfont)  

  # plot    
  plt.plot(maturities, x, color=line_color, linestyle=line_style, lw=2)  
  plt.grid(True)

  if legends is not None:
    plt.legend(legends, frameon=False, loc='upper center', ncol=len(legends))
    
  # Use matplotlib's fill_between() call to create error bars.  
  # Use the dark blue "#3F5D7D" as a nice fill color.  
  if se is not None:
    plt.fill_between(maturities, (x - se_band*se),  (x + se_band*se), alpha=alpha, color=line_color, label='_nolegend_')  
    
  if se2 is not None:
    plt.fill_between(maturities, (x - se_band*se2),  (x + se_band*se2), alpha=alpha, color=line_color2, hatch=hatch, label='_nolegend_')#"#7D5F3F")  
    
  # Finally, save the figure as a PDF, and show in log.  
  if len(filename):
    plt.savefig(filename + '.pdf', bbox_inches="tight")  
    print(filename + '.pdf')
  if show:
    # plt.show()  
    sv.save_data([x, se, filename, title], filename)
    
    

rec_start_dates = []
rec_end_dates = []
def add_rec_bars():
  global rec_start_dates, rec_end_dates
  
  # load recession data once
  if len(rec_start_dates) == 0: 
    import pandas_datareader.data as web  # module for reading datasets directly from the web
    rec = web.DataReader('USREC', 'fred', start=1972)
    rec['LAG'] = rec.USREC.shift(1)
    rec['LEAD'] = rec.USREC.shift(-1)
    rec_start_dates = rec.query('USREC==1 & LAG==0').index
    rec_end_dates = rec.query('USREC==1 & LEAD==0').index
    
  # plot recession bars
  for b, e in zip(rec_start_dates, rec_end_dates):
    plt.axvspan(b, e, facecolor='#d62728', alpha=0.10, label='_nolegend_')
              
                
def timeseries_plot(dates, x, se=None, filename='', title='', alpha=.2, line_color='', 
                    legends=None, show=True, newfig=True, recession_bars=True):
  if newfig:
    # You typically want your plot to be ~1.33x wider than tall.  
    # Common sizes: (10, 7.5) and (12, 9)  
    plt.figure(figsize=fig_size_quarter_wide)  
    
  # Remove the plot frame lines. They are unnecessary chartjunk.  
  ax = plt.gca() #subplot(111)  
  ax.spines["top"].set_visible(False)  
  ax.spines["right"].set_visible(False)  
    
  # Ensure that the axis ticks only show up on the bottom and left of the plot.  
  # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
  ax.get_xaxis().tick_bottom()  
  ax.get_yaxis().tick_left()  
    
  # Limit the range of the plot to only where the data is.  
  # Avoid unnecessary whitespace.  
  plt.xlim(dates[0], dates[-1])  

  # title (y label)
  plt.ylabel(title, **axfont)  
  #plt.title("Title", fontsize=22)  

  # plot    
  if line_color == '':
    plt.plot(dates, x, lw=1)
  else:
    plt.plot(dates, x, color=line_color, lw=1)

  # recession bars
  if recession_bars:
    add_rec_bars()
    
  # grid and legends  
  plt.grid(True)
  if legends is not None:
    plt.legend(legends, frameon=False, loc='upper center', ncol=len(legends))
    
  # Use matplotlib's fill_between() call to create error bars.  
  # Use the dark blue "#3F5D7D" as a nice fill color.  
  if se is not None:
    plt.fill_between(dates, (x - 2*se),  (x + 2*se), alpha=alpha, label='_nolegend_')  
    
  # Finally, save the figure as a PDF, and show in log. 
  if len(filename):
    plt.savefig(filename + '.pdf', bbox_inches="tight")  
    print(filename + '.pdf')
    sv.save_data([dates, x, se, filename, title], filename)
  # if show:
  #   plt.show()  



