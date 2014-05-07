import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
import datetime 
import time   

py.sign_in("etpinard", "a35m7g6el5")
stream_id = "81dygs4lct"

data = Data([Scatter(x=[],
                     y=[],
                     mode='lines+markers',
                     stream= Stream(token=stream_id,
                                    maxpoints=80))]) 

layout = Layout(title='Time Series',
                showlegend=False)

fig = Figure(data=data, layout=layout)

unique_url = py.plot(fig, filename='real-time-time-series')

s = py.Stream(stream_id)

s.open()  # open stream

print 'open stream'

i = 1
N = 30  # number of points to be plotted

while i<N:
    print i, 'write to stream'
    i += 1   # add to counter

    my_x = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # current time!
    my_y = (np.random.randn(1))[0]     # some random numbers
    
    s.write(dict(x=my_x,y=my_y))  # N.B. write to Plotly stream! 
                                  #  Send numbers to append current list
                                  #  Send list to overwrites existing list
            
    time.sleep(0.08)  # plot a point every 80 ms, for smoother plotting
    
s.close()  # close the stream when done plotting 
