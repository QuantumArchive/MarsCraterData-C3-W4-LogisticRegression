"""
Created on Tue June 27 14:34:31 2016

@author: Chris
"""
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats

#from IPython.display import display
get_ipython().magic(u'matplotlib inline')

#bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

#Set Pandas to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set Pandas to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

#data here will act as the data frame containing the Mars crater data
data = pandas.read_csv('C:\\Coursera\\marscrater_pds.csv', low_memory=False)

#convert the latitude and diameter columns to numeric and ejecta morphology is categorical
data['LATITUDE_CIRCLE_IMAGE'] = pandas.to_numeric(data['LATITUDE_CIRCLE_IMAGE'])
data['DIAM_CIRCLE_IMAGE'] = pandas.to_numeric(data['DIAM_CIRCLE_IMAGE'])
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].astype('category')

#Any crater with no designated morphology will be replaced with NaN
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].replace(' ',numpy.NaN)

#Here we will subset out craters with the three ejecta morphologies we are interested in
morphofinterest = ['Rd','SLEPS','SLERS']
data = data.loc[data['MORPHOLOGY_EJECTA_1'].isin(morphofinterest)]
data.head(5)

#We now center the data
data['CENTERED_LATITUDE'] = (data['LATITUDE_CIRCLE_IMAGE'] - data['LATITUDE_CIRCLE_IMAGE'].mean())
data['CENTERED_LONGITUDE'] = (data['LONGITUDE_CIRCLE_IMAGE'] - data['LONGITUDE_CIRCLE_IMAGE'].mean())

#We now look at our data now that we've extracted the data we wish to use
data[['LATITUDE_CIRCLE_IMAGE','LONGITUDE_CIRCLE_IMAGE','CENTERED_LATITUDE','CENTERED_LONGITUDE',
      'NUMBER_LAYERS','DEPTH_RIMFLOOR_TOPOG']].describe()

#Because of the bug in seaborn plotting, we now extract the data from the original data frame as arrays and make a new data frame
latitude = numpy.array(data['LATITUDE_CIRCLE_IMAGE'])
diameter = numpy.array(data['DIAM_CIRCLE_IMAGE'])
morphology = numpy.array(data['MORPHOLOGY_EJECTA_1'])
latitudecenter = numpy.array(data['CENTERED_LATITUDE'])
longitude = numpy.array(data['LONGITUDE_CIRCLE_IMAGE'])
depth = numpy.array(data['DEPTH_RIMFLOOR_TOPOG'])
layers = numpy.array(data['NUMBER_LAYERS'])
longitudecenter = numpy.array(data['CENTERED_LONGITUDE'])
data2 = pandas.DataFrame({'LATITUDE':latitude,
                          'DIAMETER':diameter,
                          'MORPHOLOGY_EJECTA_1':morphology,
                          'CENTERED_LATITUDE':latitudecenter,
                          'LONGITUDE':longitude,
                          'CENTERED_LONGITUDE':longitudecenter,
                          'DEPTH':depth,
                          'NUMBER_LAYERS':layers})

#Because we need a binary response and binary explanatory variable, we will split the geological latitude between the poles
#and the equator. Additionally, for our response variable, we know that from our previous study, craters with a diameter between
#1 to 1.18 (first quartile) and 1.18 to 1.53 (second quartiles) were not statistically significant enough to count as different
#(p value of 0.24), however, they were dictinct when compared to all other craters above this diameter, so we'll categorize our
#response variable as 0 for small and 1 for large.

def georegion(x):
    if x <= -30:
        return 0
    elif x <= 30:
        return 1
    else:
        return 0
    
def cratersize(x):
    if x <= 1.53:
        return 0
    else:
        return 1

#In our original model, we were only interested in craters with three kinds of ejecta morphology. Craters with Rd (radial) 
#morphology had no layers and those labeled SLERS (single layer ejecta rampart sinuous) and SLEPS (single layer ejecta pancake
#sinuous). Those craters with Rd morphology had 0 layers and the others had at least one layer, so we'll code for this so that
#0 represents no layers and anything else will have layers.

def layers(x):
    if x == 0:
        return 0
    else:
        return 1
    
#Next, we know that crater depth can be 0 or greater, so we'll also code our explanatory variable of crater depth as being 0
#as having effectively a very shallow depth and anything else as having depth

def depth(x):
    if x == 0:
        return 0
    else:
        return 1

#We now apply these equations to our variables to get new columns

#Note: we don't include longitude as our linear regression model showed no strong association
#between it and the crater diameter

data2['LATITUDE_BIN'] = data2['LATITUDE'].apply(lambda x: georegion(x))
data2['CRATER_SIZE_BIN'] = data2['DIAMETER'].apply(lambda x: cratersize(x))
data2['NUMBER_LAYERS_BIN'] = data2['NUMBER_LAYERS'].apply(lambda x: layers(x))
data2['DEPTH_BIN'] = data2['DEPTH'].apply(lambda x: depth(x))
data2.head(5)

#now we'll look at our logistic regression just using our primary variable

print('Modelling between crater size and latitude bin')
model1 = smf.logit(formula='CRATER_SIZE_BIN ~ LATITUDE_BIN',data=data2).fit()
print(model1.summary())
print('Odds Ratios')
print(numpy.exp(model1.params))

#odds ratios with 95% confidence intervals
params = model1.params
conf = model1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
numpy.exp(conf)

print('Modelling between crater size and latitude bin and layers')
model2 = smf.logit(formula='CRATER_SIZE_BIN ~ LATITUDE_BIN + NUMBER_LAYERS_BIN',data=data2).fit()
print(model2.summary())
print('Odds Ratios')
print(numpy.exp(model2.params))

#odds ratios with 95% confidence intervals
params = model2.params
conf = model2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
numpy.exp(conf)

print('Modelling between crater size and latitude bin and layers')
model3 = smf.logit(formula='CRATER_SIZE_BIN ~ LATITUDE_BIN + NUMBER_LAYERS_BIN + DEPTH_BIN',data=data2).fit()
print(model3.summary())
print('Odds Ratios')
print(numpy.exp(model3.params))

#odds ratios with 95% confidence intervals
params = model3.params
conf = model3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI','Upper CI','OR']
numpy.exp(conf)