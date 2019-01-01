# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:09:56 2018

@author: teo
"""

#IMPORT LIBRARIES
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


import plotly
plotly.offline.init_notebook_mode()
import scipy.stats as stats
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()
pd.options.display.float_format = '{:.0f}'.format
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()

states=pd.read_csv("C:\\Users\\teo\\Downloads\\european-union\\states.csv",header=0,encoding='iso-8859-1')

data=pd.read_csv("C:\\Users\\teo\\Downloads\\european-union\\states.csv",header=0,encoding='iso-8859-1')

#check for nulls
data_list=[states]
for dataset in data_list:
 print("+++++++++++++++++++++++++++")
 print(pd.isnull(dataset).sum() >0)
 print("+++++++++++++++++++++++++++")

# =============================================================================
#                       Preproccessing and Analysis
# =============================================================================

#fill nan values
states['European Union']=states['European Union'].fillna('Not in eu')
states['Council Votes']=states['Council Votes'].fillna(0)
states['European Parliament Seats']=states['European Parliament Seats'].fillna(0)
states['European Free Trade Agreement']=states['European Free Trade Agreement'].fillna('Not Member')
states['European Single Market']=states['European Single Market'].fillna('Not Member')
states['European Monetary Union']=states['European Monetary Union'].fillna('Not Member')
states['Accession Year']=states['Accession Year'].fillna(0)
states['GDP per capita ($, millions)']=states['GDP per capita ($, millions)'].fillna(0)
states['GDP ($, millions)']=states['GDP ($, millions)'].fillna(0)

#create copy of original dataframe to help us if we manipulate too much the original
states_c=states.copy()

states_corr=states.copy()


#basic mapping for most columnds on 2nd dataframe, we ll need them mostly for maps
mymap = {'Not Member':0, 'Member':1, 'Candidate':2, 'Not Applicable':3}
states_corr = states_corr.applymap(lambda s: mymap.get(s) if s in mymap else s)

# Correlations  and statistical analysis
corr=states_corr.corr()

mask=np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True

f,ax = plt.subplots(figsize=(9,9))
cmap=sns.diverging_palette(240, 14, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, ax=ax, square=True)
plt.title("Correlation Matrix")
plt.show()

plt.figure()

north_countries=states.loc[[7,8,9,17,19,14,25,33]]
western_countries=states.loc[[1,2,10,11,15,18,20,24,34,36]]
eastern_countries=states.loc[[0,3,4,6,13,21,23,26,28,29,30,31]]
south_countries=states.loc[[0,12,16,22,27,5,32,35]]

north_gpa   = north_countries.loc[:,['GDP ($, millions)']]
north_capital= north_countries.loc[:,['GDP per capita ($, millions)']]


western_gpa   = western_countries.loc[:,['GDP ($, millions)']]
western_capital= western_countries.loc[:,['GDP per capita ($, millions)']]

eastern_gpa   = eastern_countries.loc[:,['GDP ($, millions)']]
eastern_capital= eastern_countries.loc[:,['GDP per capita ($, millions)']]

south_gpa   = south_countries.loc[:,['GDP ($, millions)']]
south_capital= south_countries.loc[:,['GDP per capita ($, millions)']]


k,p=stats.mstats.normaltest(north_gpa) 

# we use p value to test if the variable is normal or not
if p<0.05:
  print ('north_gpa is not normal')
else:
  print ('north_gpa is normal')
k,p=stats.mstats.normaltest(western_gpa)
if p<0.05:
  print ('western_gpa is not normal')
else:
  print ('western_gpa is normal')
k,p=stats.mstats.normaltest(eastern_gpa)
if p<0.05:
  print ('eastern_gpa is not normal')
else:
  print ('eastern_gpa is normal')
k,p=stats.mstats.normaltest(south_gpa)
if p<0.05:
  print ('south_gpa is not normal')
else:
  print ('south_gpa is normal')
  
k,p=stats.mstats.normaltest(north_capital)

# we use p value to test if the variable is normal or not
if p<0.05:
  print ('north_capital is not normal')
else:
  print ('north_capital is normal')
k,p=stats.mstats.normaltest(western_capital)
if p<0.05:
  print ('western_capital is not normal')
else:
  print ('western_capital is normal')
k,p=stats.mstats.normaltest(eastern_capital)
if p<0.05:
  print ('eastern_capital is not normal')
else:
  print ('eastern_capital is normal')
k,p=stats.mstats.normaltest(south_capital)
if p<0.05:
  print ('south_capitalis not normal')
else:
  print ('south_capital is normal')  


print ("shapiro gdp")
print (stats.shapiro(north_gpa))
print (stats.shapiro(western_gpa))
print (stats.shapiro(eastern_gpa))
print (stats.shapiro(south_gpa))




print ("shapiro gdp capital")
print (stats.shapiro(north_capital))
print (stats.shapiro(western_capital))
print (stats.shapiro(eastern_capital))
print (stats.shapiro(south_capital ))


print ("levene capital")
print(stats.levene(north_capital,western_capital))
print(stats.levene(north_capital,eastern_capital))
print(stats.levene(north_capital,south_capital))
print(stats.levene(eastern_capital,western_capital))
print(stats.levene(eastern_capital,north_capital))
print(stats.levene(eastern_capital,south_capital))
print(stats.levene(south_capital,western_capital))
print(stats.levene(south_capital,eastern_capital))
print(stats.levene(south_capital,north_capital))

import scipy.stats as stats

print ("kruskal")
print(stats.kruskal(north_gpa.values.tolist(), western_gpa.values.tolist(), eastern_gpa.values.tolist(), south_gpa.values.tolist()))


print ("anova")
print(stats.f_oneway(north_capital.values.tolist(), western_capital.values.tolist(), eastern_capital.values.tolist(), south_capital.values.tolist()))


print(stats.ttest_ind(north_capital.values.tolist(), south_capital.values.tolist()))
print(stats.ttest_ind(north_capital.values.tolist(), western_capital.values.tolist()))
print(stats.ttest_ind(north_capital.values.tolist(), eastern_capital.values.tolist()))
print(stats.ttest_ind(eastern_capital.values.tolist(),south_capital.values.tolist()))
print(stats.ttest_ind(western_capital.values.tolist(), eastern_capital.values.tolist()))
print(stats.ttest_ind(western_capital.values.tolist(), south_capital.values.tolist()))



# =============================================================================
#                      Joint plots for correlated variables
# =============================================================================

sns.jointplot(x='European Parliament Seats',y='Council Votes',data=states,kind='reg')
sns.plt.ticklabel_format(style='plain', axis='y',useOffset=False)



sns.jointplot(x='European Parliament Seats',y='Population',data=states,kind='reg')
sns.plt.ticklabel_format(style='plain', axis='y',useOffset=False)


sns.jointplot(x='GDP ($, millions)',y='Population',data=states,kind='reg')
sns.plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.figure()


sns.jointplot(x='Area (kmÂ²)',y='Population',data=states,kind='reg')
sns.plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.figure()

sns.jointplot(x='GDP ($, millions)',y='Council Votes',data=states,kind='reg')
sns.plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.figure()

sns.jointplot(x='European Parliament Seats',y='GDP ($, millions)',data=states,kind='reg')
sns.plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.figure()





#split dataframe to four regions for plotting of distributions


south_capital['Region']='South Region'
north_capital['Region']='North Region'
eastern_capital['Region']='Eastern Region'
western_capital['Region']='Western Region'


south_gpa['Region']='South Region'
north_gpa['Region']='North Region'
eastern_gpa['Region']='Eastern Region'
western_gpa['Region']='Western Region'


mydfs = [south_capital,north_capital,eastern_capital,western_capital]

mydfs2 = [south_gpa,north_gpa,eastern_gpa,western_gpa]
regiondf = pd.concat(mydfs)


regioncapdf = pd.concat(mydfs2)


sns.boxplot( y=regiondf["Region"], x=regiondf["GDP per capita ($, millions)"] )
sns.plt.show()
sns.violinplot( x=regioncapdf["Region"], y=regioncapdf["GDP ($, millions)"] )
sns.plt.show()


sns.distplot(states['GDP ($, millions)'])


temp={}
for col in states:
    temp[col] = states[col].value_counts()
sns.set(font_scale = 4)

plt.figure(figsize=(25,20))
plt.xticks(fontsize=24,rotation=90)
#sns.set_ylabels("Survival Probability")
sns.plt.ylabel("GDP ($, millions)")
flatui = [ "#e74c3c", "#3498db", "#2ecc71"]
#sns.set(font_scale = 4)
sns.barplot(x="Country", y="GDP ($, millions)", hue="European Union",data=states.sort_values('GDP ($, millions)',ascending=False),palette=flatui)


plt.figure(figsize=(25,10))
plt.ylabel("GDP per capita ($, millions)")
#sns.set(font_scale = 4)
plt.xticks(fontsize=24,rotation=90)
sns.barplot(x="Country", y="GDP per capita ($, millions)", hue="European Union",data=states.sort_values('GDP per capita ($, millions)',ascending=False),palette=flatui)


sns.set(font_scale = 1)

# count languages,coins etc for countplots

states['language counter'] =states.Language.str.count(', ')
states['english language counter'] =states.Language.str.count('English')
states['french language counter'] =states.Language.str.count('French')
states['german language counter'] =states.Language.str.count('German')
states['greek language counter'] =states.Language.str.count('Greek')
states['euro coin count'] =states.Currency.str.count('Euro')
states['language counter']=states['language counter']+1

df = pd.DataFrame({'Category': ['one', 'two', 'three'], 'Value': [10, 20, 5]})
states.plot("Country", "language counter", kind="barh", color=sns.color_palette("deep", 3))

#color=sns.color_palette("deep", 3)
states.plot("Country", "english language counter", kind="barh", color=sns.color_palette("YlOrRd"))
states.plot("Country", "french language counter", kind="barh", color=sns.color_palette("PuBu",10))
states.plot("Country", "german language counter", kind="barh", color=sns.color_palette("OrRd",10))


states.plot("Country", "greek language counter", kind="barh", color=sns.color_palette("GnBu_d"))
states.plot("Country", "euro coin count", kind="barh", color=sns.color_palette("Blues"))
states.sort_values('GDP ($, millions)', ascending=False, inplace = True)
data['eu_years'] = 2017- data['Accession Year']
data['eu_years'] =data['eu_years'] .fillna(0)
data.sort_values('eu_years', ascending=False, inplace = True)
plt.figure()

sns.barplot(y=data['Country'], x=data['eu_years'])


plt.figure()



plt.figure()

pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]
# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='Currency', data=states, palette=pkmn_type_colors)
 
# Rotate x-labels
plt.xticks(rotation=-45)


plt.figure()
f=states['European Single Market'].groupby(states['European Single Market']).sum()
x=states.groupby('European Single Market').size()/states['European Single Market'].count()




plt.figure()
current_palette = sns.color_palette("husl", 8)
states['Accession Year'] = states['Accession Year'].astype('int32')

states.sort_values("Accession Year", inplace=True)
states = states[states['Accession Year'] > 0]

with sns.axes_style('white'):
    g = sns.factorplot("Accession Year", data=states, aspect=3.0,
                       kind="count", palette=current_palette)
    g.set_xticklabels(step=1, rotation=30 )
    


plt.figure()
#
# =============================================================================
#                           Piecharts and mapping-computations for piecharts
# =============================================================================
# 
labels = x.index.values.tolist()
myRoundedList = [ round(elem, 2) for elem in x.values.tolist() ]
sizes = myRoundedList 
colors = [ 'lightskyblue','yellowgreen']
explode = (0.1, 0)  
 

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.title('Perecentage of European Countries EU Membership')
plt.show()

plt.figure()

names='Member', 'Candidate', 'Not Applicable'
size=[(states['European Monetary Union'].str.count('Member').sum()/38)*100,(states['European Monetary Union'].str.count('Candidate').sum()/38)*100,(states['European Monetary Union'].str.count('Not Applicable').sum()/38)*100]
 

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(size, labels=names, colors=['red','green','blue'])
plt.legend( loc = 'best', labels=['%s, %1.1f %%' % (l, s) for l, s in zip(names, size)])
p=plt.gcf()
plt.title('Percentage of European Countries Monetary Union Status')
p.gca().add_artist(my_circle)
plt.show()




plt.figure()



plt.title('European Free Trade Agreement Status Count plot')
states_c['European Free Trade Agreement'].value_counts().plot(kind='barh')
plt.figure()
plt.title('European Single Market Status Count plot')
states_c['European Single Market'].value_counts().plot(kind='barh')
plt.figure()


plt.figure()



sns.countplot(y="European Single Market", hue="European Monetary Union", data=states);


plt.figure()

f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="European Monetary Union", data=states, color="c");

plt.figure()

states.reset_index(inplace=True)

states_c['index'] = states_c.index
new = states[['index', 'Country', 'Council Votes','European Parliament Seats']].copy()
new2 = states[['index', 'Country', 'Area (kmÂ²)']].copy()
new3 = states_c[['index', 'Country', 'European Monetary Union','European Single Market','European Free Trade Agreement']].copy()
new4 = states_c[['index', 'Country', 'European Union']].copy()
df1 = pd.melt(new, id_vars=['index','Country']).sort_values(['variable','value'])
df2 = pd.melt(new2, id_vars=['index','Country']).sort_values(['variable','value'])
df3 = pd.melt(new3, id_vars=['index','Country']).sort_values(['variable','value'])

#df1.loc[(df1['column_name'] == some_value) & df1['other_column'].isin(some_values)]
#
#sns.barplot(x="Country", y="Council Votes", hue='European Parliament Seats', data=states)
#
sns.barplot(x='Country', y='value', hue='variable', data=df1)
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.title('Council Votes vs European Parliament Seats');

states.sort_values(['Population'],ascending=False)

plt.figure()


sns.barplot(x='Country', y='value', hue='variable', data=df2)
plt.xticks(rotation=90)
plt.ylabel('Size')
plt.title('Country Area Comparison');


plt.figure()

mymap = {'Not Member':0, 'Member':3, 'Candidate':2, 'Not Applicable':4}
df3 = df3.applymap(lambda s: mymap.get(s) if s in mymap else s)

mymap2  = {'Not Member':0.0, 'Member':0.3, 'Candidate':0.2, 'Not Applicable':0.4}

new4= new4.applymap(lambda s: mymap2.get(s) if s in mymap2 else s)
df4 = pd.melt(new4, id_vars=['index','Country']).sort_values(['variable','value'])

sns.barplot(x='Country', y='value', hue='variable', data=df3)
plt.xticks(rotation=90)
plt.ylabel('Returns')
plt.yticks(np.arange(5),('', '', 'Candidate', 'Member', 'Not Applicable'))

plt.title('Country Memberships and Agreements');




plt.figure()

sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(20, 25))
ax.set_title('Country Population-Gdp Plot',fontsize=24)
ax.set_xlabel('GDP($,millions-Red)-Population(Blue)',fontsize=24)
ax.set_ylabel('Country',fontsize=24)

sns.set_color_codes("pastel")
sns.barplot(x="Population", y="Country", data=states,
            label="Total", color="b")

sns.set_color_codes("muted")
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)

sns.barplot(x="GDP ($, millions)", y="Country", data=states,
            label="Alcohol-involved", color="r")
#
# =============================================================================
#                           Choropleth Maps
# =============================================================================

countt=states['Country'].values.tolist()
data = dict(type='choropleth',
locations = states_c['Country'],
locationmode = 'country names', z = states_c['Population Density'],
text = states_c['Country'], colorbar = {'title':'pop density per 1 khm.'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
reversescale = False)


layout = dict(title='Population Density Map',
geo = dict(showframe = True, projection={'type':'Mercator'}))



choromap = go.Figure(data = [data], layout = layout)
plotly.offline.plot(choromap, filename='append plot',validate=False)

data2 = dict(type='choropleth',
locations = states_c['Country'],
locationmode = 'country names', z = states_c['Population'],
text = states_c['Country'], colorbar = {'title':'Population climax'},
        colorscale='Rainbow',    
reversescale = False)


layout2 = dict(title='Population Map',
geo2 = dict(showframe = True, projection={'type':'Mercator'}))



choromap2 = go.Figure(data = [data2], layout = layout2)
plotly.offline.plot(choromap2,filename='append plot2', validate=False)
df3['value'] = df3['value'].apply(lambda x: x*0.1)



trade=pd.DataFrame()
market=pd.DataFrame()
union=pd.DataFrame()
union=df3.loc[df3['variable'] == 'European Monetary Union']
market=df3.loc[df3['variable'] == 'European Single Market']
trade=df3.loc[df3['variable'] == 'European Free Trade Agreement']
scl = [[0.0, 'rgb(255, 0, 255)'],[0.2, 'rgb(153, 102, 51)'],[0.3, 'rgb(102, 255, 102)'],\
            [0.4, 'rgb(26, 0, 0)'],[1.0, 'rgb(255, 102, 102)']]
df4=pd.DataFrame()







unionscl = [[0.0, 'rgb(255, 0, 255)'],[0.3, 'rgb(102, 255, 102)'],[1.0, 'rgb(255, 102, 102)']]
unionscl2 = [[0.0, 'rgb(255, 0, 255)'],[1.0, 'rgb(255, 102, 102)']]

data3 = dict(type='choropleth',
locations = trade['Country'],
locationmode = 'country names', z = trade['value'],
text = trade['Country'], colorbar = {'title':'European Free Trade Members'},
        colorscale=unionscl,    
reversescale = False)


layout3 = dict(title='European Free Trade Members Map',
#font=dict(family='Courier New, monospace', size=20, color='#7f7f7f'),             
geo3 = dict(showframe = True, projection={'type':'Mercator'}))



choromap3 = go.Figure(data = [data3], layout = layout3)
plotly.offline.plot(choromap3,filename='append plot3', validate=False)





data4 = dict(type='choropleth',
locations = market['Country'],
locationmode = 'country names', z = market['value'],
text = market['Country'], colorbar = {'title':'European Single Market Members'},
        colorscale='Rainbow',    
reversescale = False)


layout4 = dict(title='European Single Market Members Map',
geo4 = dict(showframe = True, projection={'type':'Mercator'}))



choromap4 = go.Figure(data = [data4], layout = layout4)
plotly.offline.plot(choromap4,filename='append plot4', validate=False)



unionscl = [[0.0, 'rgb(255, 0, 255)'],[0.3, 'rgb(102, 255, 102)'],[1.0, 'rgb(255, 102, 102)']]


data5 = dict(type='choropleth',
locations = union['Country'],
locationmode = 'country names', z = union['value'],
text = union['Country'], colorbar = {'title':'European Monetary Union Members'},
        colorscale='Rainbow',    
reversescale = False)


layout5 = dict(title='European Monetary Union Map',
geo5 = dict(showframe = True, projection={'type':'Mercator'}))



choromap5 = go.Figure(data = [data5], layout = layout5)
plotly.offline.plot(choromap5,filename='append plot5', validate=False)








data6 = dict(type='choropleth',
locations = states_c['Country'],
locationmode = 'country names', z =new4['European Union'],
text = states_c['Country'], colorbar = {'title':'European  Union Members'},
        colorscale='Viridis',    
reversescale = False)


layout6 = dict(title='European Union Map',
geo6 = dict(showframe = True, projection={'type':'Mercator'}))



choromap6 = go.Figure(data = [data6], layout = layout6)
plotly.offline.plot(choromap6,filename='append plot6', validate=False)
















