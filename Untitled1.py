
# coding: utf-8

# In[9]:

from IPython.html.widgets import*
import numpy as np
from matplotlib import pyplot as plt
t=np.arange(0.0,1.0,0.01)

def pltsin(f):
    plt.plot(t,np.sin(2*np.pi*t*f))
    plt.show()

interact(pltsin,f=(1,10,0.1))


# In[13]:

import numpy as np
from matplotlib import pyplot as plt

x=np.random.rand(15)
y=x+np.random.rand(15)
z=x+np.random.rand(15)
z=z*z

plt.scatter(x, y, s=z*2000, c=x, cmap="BuPu_r", alpha=.4, edgecolors="grey", linewidth=2)


# In[16]:

from IPython.html.widgets import*
t=np.arange(0.0,1.0,0.01)

def pltsin(coef):
    x=[i for i in range(10)]
    y=[i*coef+np.random.uniform() for i in range(10)]
    plt.scatter(x,y)
    plt.ylim(0,100)
    plt.xlim(0,10)
    plt.show()
    
interact(pltsin, coef={'one':1, 'ten':10,'point one':0.1})


# In[61]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.DataFrame({'x':range(1,101),'y':np.random.randn(100)*15+range(1,101)})
plt.plot('x','y',data=df,linestyle='none',marker='v')
plt.annotate('This point is interesting!', xy=(21,44), xytext=(0,80),
arrowprops=dict(facecolor='red',shrink=.05)
)
plt.show()


# In[36]:

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x=np.random.rand(80)-0.5
y=x+np.random.rand(80)
z=x+np.random.rand(80)
df=pd.DataFrame({'x':x, 'y':y, 'z':z})
sns.lmplot(x='x',y='y',data=df,fit_reg=False,hue='x',legend=False,palette="Blues")
sns.lmplot(x='x',y='y',data=df,fit_reg=False,hue='x',legend=False,palette="Blues_r")
plt.show()


# In[69]:

import matplotlib.pyplot as plt
import numpy as np
x=np.random.rand(15)
y=x+np.random.rand(15)
z=x+np.random.rand(15)
z=z*z

plt.scatter(x, y, s=z*2000, c=x, cmap="BuPu", alpha=.4, edgecolors="grey", linewidth=2)
plt.scatter(x, y, s=z*2000, c=x, cmap="BuPu_r", alpha=.4, edgecolors="grey", linewidth=2)
plt.scatter(x, y, s=z*2000, c=x, cmap="spring", alpha=.4, edgecolors="grey", linewidth=2)
plt.show()


# In[71]:

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
text=("Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas Datascience Wordcloud Spider Radar Parallel Alpha Color Brewer Density Scatter Barplot Barplot Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie Time-series Wordcloud Wordcloud Sankey Bubble")
wordcloud=WordCloud(width=480,height=480,margin=0).generate(text)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.margins(x=0,y=0)
plt.show()


# In[ ]:



