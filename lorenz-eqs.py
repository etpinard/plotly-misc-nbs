
# coding: utf-8

# In[1]:

import numpy as np
from scipy import integrate
import time


# In[2]:

import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *

stream_ids = tls.get_credentials_file()['stream_ids']
stream_ids


# In[3]:

def lorentz_deriv((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# In[4]:

# Number individual trajectories to integrate
N_trajectories = 1

# Choose random starting points, uniformly distributed from -15 to 15
np.random.seed(1)
x0 = -15 + 30 * np.random.random((N_trajectories, 3))


# In[7]:

stream1 = dict(
    token=stream_ids[0],
    maxpoints=3000   
)

stream2 = dict(
        token=stream_ids[1]
)

trace1 = dict(
    type='scatter3d',
    x=[0],
    y=[0],
    z=[0],
    mode='lines',
    stream=stream1
)

trace2 = dict(
    type='scatter3d',
    x=[0],
    y=[0],
    z=[0],
    mode='markers',
    marker=dict(
        color="#1f77b4",
        size=12,
        symbol='circle'
    ),
    stream=stream2
)

data = [trace1, trace2]


# In[8]:

layout = dict(
    title='Lorenz Attractor',
    scene=dict(
        xaxis=dict(
            autorange=False,
            range=[-25,25]
        ),
        yaxis=dict(
            autorange=False,
            range=[-35,35]
        ),
        zaxis=dict(
            autorange=False,
            range=[0,55]
        )
    ),
    margin=dict(
        l=0,
        r=0,
        t=80,
        b=0
   )
)


# In[9]:

fig = dict(data=data, layout=layout)


# In[10]:

py.plot(fig, validate=False, filename='lorenz-eqs-stream')


# In[12]:

s1 = py.Stream(stream_ids[0])
s1.open()

s2 = py.Stream(stream_ids[1])
s2.open()


# In[13]:

np.random.seed(1)
x0 = -15 + 30 * np.random.random((1, 3))[0]

while True:
    
    t = np.linspace(0, 4, 1000)
    X_t = integrate.odeint(lorentz_deriv, x0, t)
    
    for x_t in X_t:
        
        s_data1 = dict(
            type='scatter3d',
            x=x_t[0],
            y=x_t[1],
            z=x_t[2]
        )

        s_data2 = dict(
            type='scatter3d',
            x=[x_t[0]],
            y=[x_t[1]],
            z=[x_t[2]]
        )

        s1.write(s_data1, validate=False)
        s2.write(s_data2, validate=False)

        time.sleep(0.05)
        
    x0 = X_t[-1,:]

s.close()


# In[ ]:



