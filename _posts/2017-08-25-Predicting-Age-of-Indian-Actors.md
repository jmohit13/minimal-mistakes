---
title: "Predicting Age of Indian Actors"
categories:
  - Data-Science
tags:
  - Machine Learning
  - python
---

### This is a data science problem from Analytcs Vidhya where we have to predict the age Indian Actors.

This [dataset](https://datahack.analyticsvidhya.com/contest/practice-problem-age-detection/){:target="_blank"} contains images of 100 Indian actors collected from more than 100 videos. To simplify the problem, it has been converted into a mutliclass problem with 3 classes Young, Middle and Old. The original dataset can be downloaded from [here](http://cvit.iiit.ac.in/projects/IMFDB/){:target="_blank"} as well

The data attributes are
<pre>
	* ID - Unique ID of image
	* Class – Age bin of person in image
</pre>

Let`s load the libraries


{% highlight python linenos %}
# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn import over_sampling as os
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
{% endhighlight %}

Now load the data 
{% highlight python linenos %}
# load data
train = pd.read_csv('../data/Age_detection_analytics_vidhya/train.csv')
test = pd.read_csv('../data/Age_detection_analytics_vidhya/test.csv')
train.info()
train.head()
{% endhighlight %}

The dataset has 2 columns one the image ID and Class variable. Let`s get started with our exploratory data analysis. Generally the number of actors would be less in 'OLD' & 'YOUNG' age. 

{% highlight python linenos %}
X = train['ID'].values
y = train['Class'].values
X = X.reshape(len(X), 1)

def class_info(data):
    '''
        This functions outputs the class percentage
    '''
    unique, counts = np.unique(data, return_counts=True)
    class_dict = dict(zip(unique, counts))
    total = sum([y for x,y in class_dict.items()])
    for x,y in class_dict.items():
        print("{} : {:.2f}%".format(x,y*100/total))
        
class_info(y)
{% endhighlight %}

We find 'MIDDLE' class as the dominant class here and we have to fight with class imbalance




