
# coding: utf-8

## Machine Learning

### classifier comparison using Plotly

# this got an amazing 76 retweets:
# 
# https://twitter.com/SahaSurya/status/470552623041761280
# 
# let's ipython notebook this shit.
# 
# here's the code:
# 
# http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html

# <hr>
# <br>
# 
# First import a few modules:

# In[1]:

import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA


# In[2]:

import plotly.plotly as py
import plotly.tools as tls


# In[3]:

from plotly.graph_objs import Figure, Data, Layout
from plotly.graph_objs import Scatter, Contour
from plotly.graph_objs import Marker, Contours, Font
from plotly.graph_objs import XAxis, YAxis, Annotation, Annotations


# In[4]:

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]


#### 1. Original matplotlib version

# In[5]:

get_ipython().magic(u'matplotlib inline')

figure = pl.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = pl.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
#pl.show()


#### 2. Plotly version

# In[5]:

#
cm_bright = ['#FF0000', '#0000FF']

# Function to make Scatter graph object to plot the datasets' pts
def make_Scatter(sbplt_in,x_in,y_in,color_in,opacity_in):
    return Scatter(
        x= x_in,
        y= y_in,
        mode='markers',
        marker= Marker(
            color= color_in,
            opacity= opacity_in),
        name='',
        xaxis= 'x{}'.format(sbplt_in),
        yaxis= 'y{}'.format(sbplt_in))


# In[6]:

#
cm_name = 'RdBu'

# Function to make Contour graph object to plot the 'decision boundary'
def make_Contour(sbplt_in,x_in,y_in,Z_in):
    return Contour(
        x= x_in,
        y= y_in,
        z= Z_in,
        scl= cm_name,
        showscale=False,
        reversescl=False,
        contours= Contours(showlines=False),
        #opacity=0.8,
        xaxis= 'x{}'.format(sbplt_in),
        yaxis= 'y{}'.format(sbplt_in))


# In[7]:

# Some style options for all x- and y-axes
axis_style = dict(
    ticks='',
    showticklabels=False,
    showline=True,
    mirror=True,
    showgrid=False,
    zeroline=False)

# Function to make XAxis graph object
def make_XAxis(x_in):
    xaxis = XAxis(range=[x_in.min(),x_in.max()])
    xaxis.update(axis_style)
    return xaxis

# Function to make YAxis graph object
def make_YAxis(y_in):
    yaxis = YAxis(range=[y_in.min(),y_in.max()])
    yaxis.update(axis_style)
    return yaxis


# In[8]:

#
def make_score_anno(sbplt_in,x_in,y_in,score):
    return Annotation(
        x= x_in.max() - 0.95,   # had to tweak these from
        y= y_in.min() + 0,      #  from original
        text= ('%.2f' % score).lstrip('0'),
        align='right',
        font= Font(size=15),
        showarrow=False,
        xref= 'x{}'.format(sbplt_in),
        yref= 'y{}'.format(sbplt_in))

#
def make_sbplt_anno(sbplt_in,x_in,y_in,name):
    return Annotation(
        x= np.mean(x_in),
        y= y_in[-1],
        text= name, 
        xanchor='center',
        align='center',
        font= Font(size=14),
        showarrow=False,
        xref= 'x{}'.format(sbplt_in),
        yref= 'y{}'.format(sbplt_in))


# Generate figure object with subplot layout:

# In[9]:

#
figure = tls.get_subplots(
    rows=len(datasets),
    columns=len(classifiers)+1,
    horizontal_spacing=0.01,
    vertical_spacing=0.05,
    print_grid=True)


# Add a few style options:

# In[10]:

#
figure['layout'].update(showlegend=False,
                        hovermode='closest',
                        autosize=False,
                        width=1472,
                        height=490)

# 
figure['layout'].update(title='Machine Learning classifier comparison',
                        font= Font(family="Open Sans, sans-serif"))
# Init. 
figure['layout']['annotations'] = Annotations([])


# Loop through the datasets and the classifiers to fill in the figure object:

# In[11]:

i = 1   #

# iterate over datasets (in reverse order, to match original layout)
for ds in datasets[::-1]:
    
    # preprocess dataset, split into training and test part
    X, _y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, _y, test_size=.4)
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    x = np.arange(x_min, x_max, h)
    y = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(x,y)
    
    #
    cm_train = [cm_bright[yy_train] for yy_train in y_train]
    cm_test = [cm_bright[yy_test] for yy_test in y_test]
    
    #
    figure['data'] += [make_Scatter(i,X_train[:, 0],X_train[:, 1],cm_train,1)]
    figure['data'] += [make_Scatter(i,X_test[:, 0],X_test[:, 1],cm_test,0.6)]
    
    #
    figure['layout'].update({'xaxis{}'.format(i): make_XAxis(x)})
    figure['layout'].update({'yaxis{}'.format(i): make_XAxis(y)})
      
    i += 1   #
    
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        
        # 
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        figure['data'] += [make_Contour(i,x,y,Z)]
        
        # Plot also the training points and testing points
        figure['data'] += [make_Scatter(i,X_train[:, 0],X_train[:, 1],cm_train,1)]
        figure['data'] += [make_Scatter(i,X_test[:, 0],X_test[:, 1],cm_test,0.6)]
        
        # 
        figure['layout'].update({'xaxis{}'.format(i): make_XAxis(x)})
        figure['layout'].update({'yaxis{}'.format(i): make_YAxis(y)})
        
        #
        figure['layout']['annotations'] += [make_score_anno(i,x,y,score)]
        
        # 
        if i>=22:
            figure['layout']['annotations'] += [make_sbplt_anno(i,x,y,name)]
        
        i += 1   # 
        


# In[12]:

print figure['layout'].to_string()


# In[18]:

import json

class NumpyAwareJSONEncoder(json.JSONEncoder):
     def default(self, obj):
         if isinstance(obj, np.ndarray):
                 return obj.tolist()
         return json.JSONEncoder.default(self, obj)

with open('figure.json', 'w') as outfile:
    json.dump(figure, outfile, cls=NumpyAwareJSONEncoder)


# In[20]:

#cat figure.json


# In[14]:

get_ipython().run_cell_magic(u'timeit', u'', u"\n#\npy.iplot(figure, filename='ml-classifier-comp')")


# <br>
# <hr>

# <div style="float:right; \">
#     <img src="http://i.imgur.com/4vwuxdJ.png" 
#  align=right style="float:right; margin-left: 5px; margin-top: -10px" />
# </div>
# 
# <h4 style="margin-top:80px;">Got Questions or Feedback? </h4>
# 
# About <a href="https://plot.ly" target="_blank">Plotly</a>
# 
# * email: feedback@plot.ly 
# * tweet: 
# <a href="https://twitter.com/plotlygraphs" target="_blank">@plotlygraphs</a>
# 
# <h4 style="margin-top:30px;">Notebook styling ideas</h4>
# 
# Big thanks to
# 
# * <a href="http://nbviewer.ipython.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Prologue/Prologue.ipynb" target="_blank">Cam Davidson-Pilon</a>
# * <a href="http://lorenabarba.com/blog/announcing-aeropython/#.U1ULXdX1LJ4.google_plusone_share" target="_blank">Lorena A. Barba</a>
# 
# <br>

# In[11]:

# CSS styling within IPython notebook
from IPython.core.display import HTML
import urllib2
def css_styling():
    url = 'https://raw.githubusercontent.com/plotly/python-user-guide/master/custom.css'
    styles = urllib2.urlopen(url).read()
    return HTML(styles)

css_styling()


# In[11]:



