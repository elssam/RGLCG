# Import all software libraries

```python
import pandas as pd
import scipy.stats as stats
%matplotlib inline
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
```

# Read all data in a very long format


```python
data=pd.read_csv("Final.csv") #raw features
```


```python
data.head()
```


```python
REG= data.region.unique().tolist()
STA= data.status.unique().tolist()
VIS= data.visit.unique().tolist()
```


```python
L=[x[6:] for x in REG]
```

# Reading genetic data (as probesets)


```python
#integrate and test
TRY=pd.read_csv("TEMP") #all biomart genes 56, unique are 18.
TRY=TRY.T
#TRY.head()

#Rename Columns as the second row (genes)
TRY.columns=TRY.iloc[2]
later=TRY.iloc[0:3]
########################################################
#remove unwanted rows, initialy
TRY.drop(index=['Phase', 'Unnamed: 1','Unnamed: 747'],inplace=True)
#rename columns so that nan is 'column'
TRY.columns = [ x if str(x)==x else 'column' for x in TRY.columns ]
#change the second (index=1) to be 'id'
TRY.columns = [ 'id' if i==1 else x for i,x in enumerate(TRY.columns) ]
#remove rest of all unwanted columns/rows
TRY.drop(['column','Symbol'],axis=1,inplace=True)
TRY.drop('Unnamed: 2',inplace=True)
k={}
n=[]
for x in TRY.columns:
    if x in n:
        n.append(x+"-"+str(k[x]))
        k[x]+=1
    else:
        n.append(x)
        k[x]=1

TRY.columns =n
#now rename duplicate columns
#TRY.rename(columns=lambda c: d[c].pop(0) if c in d.keys() else c)
TRY.head()
```

# Read the information about probe sets and gene ids


```python
later2=later.iloc[:,7:]
later_final=later2.T
later_final["names"]= TRY.columns[:]
#later_final.to_csv("IDs.csv")
```


```python
later_final
```


```python
unique_names=later_final["Unnamed: 2"].unique()
unique_names=unique_names[1:]
```


```python
#Sava the dataset
###TRY.to_csv("Genes_65.csv")
```

# Select Genes

### Import a dataset with the absolute differenes to see which probesets are differentially expressed


```python
testy=pd.read_csv("Differences_All.txt")
```


```python
testy=testy[testy.region_baseline=="region1"]
#masky=testy.status_baseline.where([testy.status_baseline=='AD'],True)
masky=testy.status_baseline=='AD'
masky2=masky==False
```


```python
def ttesty(x):
    '''Find the Genes that were differentialy expressed'''
    m=masky
    m2=masky2
    return stats.mannwhitneyu(x[m2],x[m])[1]
#    return stats.mannwhitneyu(x.mask[m],x.mask[m2])[1]
```

### Conduct Mannwhiteney U test


```python
fo=open("new/probe_to_gene.txt","wt")
fo.writelines("Gene"+"&"+"Probe set id"+"&"+"P-value"+"\\\\"+"\n")
d={}
for i in unique_names:
    filter_col = [col for col in TRY.columns if col.startswith(i)]
    df=testy[filter_col]
    #print(df.astype(float).apply(np.var))
    print_me=df.astype(float).apply(ttesty)
    ix=print_me[print_me==min(print_me)].index[0]
    pid=later_final[later_final.names==ix].Phase[0]
    d[i]=[min(print_me),pid,ix]
    fo.writelines("\\bfseries  "+i+"&"+str(pid)+"&"+str(round(min(print_me),5))+"\\\\"+"\n")
    print(i,"&",pid,"&",round(min(print_me),5),"\\\\")#,ix)"
```

# Define the new Gene df (TRY)

### Define the new column of probsets that passes the test above


```python
#select the following columns
nw_col=[d[i][-1] for i in d]
nw_col.append("id")
#rename with the following
new_col=[i for i in d]
new_col.append("id")
```


```python
TRY.head()
```


```python
TRY=TRY[nw_col]
```


```python
TRY.columns=new_col
```


```python
TRY.head()
```

Now, do the same analysis like above:

# Testing Association (gene with region difference) with Spearman


```python
round(0.05/(90*17),7)
```


```python
0.1/(90)
```


```python
# top 20
#I AM CHEATING HERE: by using duplicate genes. Get unique ask for help

#Reshape for each region (loop through regions)
ff=pd.read_csv("new/alex.txt",header=None,names=['region','reg_id','AAL_name'])

fo=open("new/spearman_local.txt","wt")
fo.writelines("&".join('Gene,Region,Region id,Metric,r,P-value'.split(","))+"\\\\"+"\n")
for reg in REG:
    R=ff[ff.region==reg]['AAL_name'].iloc[0]
    merge1=[]
    REGION1=data[(data.region==reg)]
    #cases
    test1 = pd.merge(REGION1[REGION1.visit=="baseline"],REGION1[REGION1.visit=="followup"],on="id",suffixes=('_baseline', '_followup'))
    #Calculate the differences
    for i in ["betweencentrality","local_eff","cluster_coef"]:
        b=i+"_baseline"
        a=i+"_followup"
        test1[str(i)]=abs(test1[b]-test1[a])
    #integration of genetic data
    #integrate and test
    merge1 = pd.merge(test1,TRY,on="id")
    F=["betweencentrality","local_eff","cluster_coef"]
    G=range(1,len(TRY.columns))
    for g in G:
        for f in F:
            gi=g+15
            results=stats.spearmanr(merge1[f],merge1.iloc[:,gi])
            if results.pvalue <= 0.003:#1111111111111111: #0.1÷(90)
                print (reg[6:],R,merge1.columns[gi],f,round(results[0],4),round(results[1],6))
                line="&".join([str(a) for a in [merge1.columns[gi],R,'Region'+reg[6:],f,round(results[0],4),round(results[1],6)]])
                fo.writelines("\\bfseries  "+line+"\\\\"+"\n")
fo.close()
```

# Per Feature ASSOCIATION Analysis (QUANTILE REGRESSION)

# Heatmaps: All genes


```python
#prepare heatmap data by multiplying by 1000

def times_no(List,n=1000):
    '''This function multiplies your list entries by 10 (or provide a second argument n). You can use apply() to perform it into dataframe columns'''
    new_List = []
    for i in List:
        j = float(i)
        j=j*n
        new_List.append(j)
    return new_List
```

# Heatmaps: Genes included only


```python
#prepare for heatmap by multiplying by 1000 and only genes that has features
testt= merge1
testt=testt.iloc[:,16:]
print(testt.head())
testt_plot=testt.apply(times_no)
print(testt_plot.head())
```


```python
#without row cluster

sns.set(font_scale=1.3)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
#plt.figure(figsize=(150,60))
g=sns.clustermap(data=testt_plot,row_cluster=False,col_cluster=True,xticklabels=True,figsize=(10,10), annot=False, fmt='d',yticklabels=False, cmap="YlGnBu",)#row_colors=testt.iloc[:,12])# annot_kws={"size": 10})#,cbar_kws={10:1000,8:8000,6:6000,4:4000,2:2000})
plt.xlabel('Gene Expressions*1000', fontsize=15)

#col_colors=testt.iloc[:,12].T
###plt.savefig("gene_expression.png")
```

# Creating all datasets with regions


```python
#NOTE THE DATA WILL HAVE REDUNDANCY IN GENE EXPRESSIONS: SOLVED

data_all=pd.DataFrame()
for reg in REG:
    merge1=[]
    REGION1=data[(data.region==reg)]
    #cases
    test1 = pd.merge(REGION1[REGION1.visit=="baseline"],REGION1[REGION1.visit=="followup"],on="id",suffixes=('_baseline', '_followup'))
    #Calculate the differences
    for i in ["betweencentrality","local_eff","cluster_coef"]:
        b=i+"_baseline"
        a=i+"_followup"
        test1[str(i)]=abs(test1[b]-test1[a])
    #integration of genetic data
    #integrate and test
    merge1 = pd.merge(test1,TRY,on="id")
    data_all=data_all.append(merge1)
```


```python
#Correlations:
#plot correlations
#sns.heatmap(merge1.corr(method='spearman'), cmap="YlGnBu")
sns.set(font_scale=1.4)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
sns.clustermap(data=data_all.corr(method='spearman'),row_cluster=True,yticklabels=True, annot=True,xticklabels=True, cmap="YlGnBu", annot_kws={"size": 14},mask=False)
####plt.savefig("feature_correlation.png")
#print it

data_all.corr(method='spearman')
```


```python
data_all.head()
```


```python
data_all.shape
```

# Plot Differences for brain-wide


```python
data_all.describe()
```


```python
L=[x[6:] for x in REG]
```


```python
s=[]
for j,i in enumerate((89*' ').split(" ")):
#    print(j)
    if (j+1 in np.arange(1,len(L)+1,4)):
        s.append(j+1)
    else:
        s.append(i)
print (s)
```


```python
#plot differences
sns.set()

sns.set(style="whitegrid", palette="muted",font_scale=2.5)
plt.figure(figsize=(35,20 ))
# Draw a categorical scatterplot to show each observation
g=sns.swarmplot(x=data_all.region_baseline, y=data_all.cluster_coef
                ,dodge=True,palette=["darkblue","gold"], size=9,hue=data_all.status_baseline,)#alpha=0.8)
g.set_xticklabels(s)
l = g.legend()
l.set_title('Status')
plt.xticks(fontsize=35,)# rotation=90)
plt.yticks(fontsize=35 )
plt.xlabel("Brain Region",fontsize=40)
plt.ylabel("Clustering Coefficient Change",fontsize=40)


#plt.xticks(rotation=90)

```


```python
#plot differences
sns.set()

sns.set(style="whitegrid", palette="muted",font_scale=2.5)
plt.figure(figsize=(35,20 ))
# Draw a categorical scatterplot to show each observation
g=sns.swarmplot(x=data_all.region_baseline, y=data_all.local_eff
                ,dodge=True,palette=["darkblue","gold"], size=9,hue=data_all.status_baseline,)#alpha=0.8)
g.set_xticklabels(s)
l = g.legend()
l.set_title('Status')
plt.xticks(fontsize=35,)# rotation=90)
plt.yticks(fontsize=35 )
plt.xlabel("Brain Region",fontsize=40)
plt.ylabel("Local Efficiency Change",fontsize=40)


#plt.xticks(rotation=90)

```


```python
#plot differences
sns.set()

sns.set(style="whitegrid", palette="muted",font_scale=2.5)
plt.figure(figsize=(35,20 ))
# Draw a categorical scatterplot to show each observation
g=sns.swarmplot(x=data_all.region_baseline, y=data_all.betweencentrality
                ,dodge=True,palette=["darkblue","gold"], size=9,hue=data_all.status_baseline,)#alpha=0.8)
g.set_xticklabels(s)
l = g.legend()
l.set_title('Status')
plt.xticks(fontsize=35,)# rotation=90)
plt.yticks(fontsize=35 )
plt.xlabel("Brain Region",fontsize=40)
plt.ylabel("Betweenness Centrality Change",fontsize=40)

#plt.xticks(rotation=90)

```


```python
sns.set()

sns.set(style="whitegrid", palette="muted",font_scale=2.5)
plt.figure(figsize=(35,20 ))
# Draw a categorical scatterplot to show each observation
g=sns.swarmplot(x="region", y="betweencentrality", hue="visit",
                dodge=False,palette=["purple","limegreen"], size=9,data=data)
g.set_xticklabels(s)
l = g.legend()
l.set_title('Visit')
plt.xticks(fontsize=35,)# rotation=90)
plt.yticks(fontsize=35 )
plt.xlabel("Brain Region",fontsize=40)
plt.ylabel("Betweenness Centrality",fontsize=40)
plt.savefig("new/betweenness_bf.png")
plt.show()

sns.set()

sns.set(style="whitegrid", palette="muted",font_scale=2.5)
plt.figure(figsize=(35,20 ))
g=sns.swarmplot(x="region", y="local_eff", hue="visit",
                 dodge=False,palette=["purple","limegreen"], size=9,data=data)
g.set_xticklabels(s)
l = g.legend()
l.set_title('Visit')
plt.xticks(fontsize=35,)# rotation=90)
plt.yticks(fontsize=35 )
plt.xlabel("Brain Region",fontsize=40)
plt.ylabel("Local Efficiency",fontsize=40)
plt.savefig("new/local_eff_bf.png")
plt.show()

sns.set()

sns.set(style="whitegrid", palette="muted",font_scale=2.5)
plt.figure(figsize=(35,20 ))
g=sns.swarmplot(x="region", y="cluster_coef", hue="visit",
                 dodge=False,palette=["purple","limegreen"], size=9,data=data)

g.set_xticklabels(s)
l = g.legend()
l.set_title('Visit')
plt.xticks(fontsize=35,)# rotation=90)
plt.yticks(fontsize=35 )
plt.xlabel("Brain Region",fontsize=40)
plt.ylabel("Clustering Coefficient",fontsize=40)
plt.savefig("new/cluster_coef_bf.png")
plt.show()



```

# Regions Associations


```python
v='betweencentrality'
lc=[v,'region_baseline','id']
regcor=data_all[lc]
regcor=regcor.pivot(index='id', columns='region_baseline', values='betweencentrality')
regcor=regcor.reindex(['region'+str(i) for i in range(1,91)], axis=1)
regcor.columns=[str(i) for i in range(1,91)]
```


```python
#Correlations:
#plot correlations
#sns.heatmap(merge1.corr(method='spearman'), cmap="YlGnBu")
sns.set(font_scale=.75)
plt.figure(figsize=(30,30))
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
sns.clustermap(data=regcor.corr(method='spearman')*1000,row_cluster=True,col_cluster=True,yticklabels=True,xticklabels=True, cmap="YlGnBu",mask=False)
####plt.savefig("feature_correlation.png")
#print it

regcor.corr(method='spearman')*1000

```


```python
len()
```


```python
v='local_eff'
lc=[v,'region_baseline','id']
regcor=data_all[lc]
regcor=regcor.pivot(index='id', columns='region_baseline', values='local_eff')
regcor=regcor.reindex(['region'+str(i) for i in range(1,91)], axis=1)
regcor.columns=[str(i) for i in range(1,91)]
```


```python
#Correlations:
#plot correlations
#sns.heatmap(merge1.corr(method='spearman'), cmap="YlGnBu")
sns.set(font_scale=.75)
plt.figure(figsize=(30,30))
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
sns.clustermap(data=regcor.corr(method='spearman')*1000,row_cluster=True,col_cluster=True,yticklabels=True,xticklabels=True, cmap="YlGnBu",mask=False)
####plt.savefig("feature_correlation.png")
#print it

regcor.corr(method='spearman')*1000

```


```python
v='cluster_coef'
lc=[v,'region_baseline','id']
regcor=data_all[lc]
regcor=regcor.pivot(index='id', columns='region_baseline', values='cluster_coef')
regcor=regcor.reindex(['region'+str(i) for i in range(1,91)], axis=1)
regcor.columns=[str(i) for i in range(1,91)]
```


```python
#Correlations:
#plot correlations
#sns.heatmap(merge1.corr(method='spearman'), cmap="YlGnBu")
sns.set(font_scale=.75)
plt.figure(figsize=(30,30))
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
sns.clustermap(data=regcor.corr(method='spearman')*1000,row_cluster=True,col_cluster=True,yticklabels=True,xticklabels=True, cmap="YlGnBu",mask=False)
####plt.savefig("feature_correlation.png")
#print it

regcor.corr(method='spearman')*1000

```

### Note: did not do it per status

# Trying a new Quantile Regression Model


```python
#Here i am trying to create a dataframe of p-values and plot it as heatmap
#RE=REG[:10]
ALL_in=pd.DataFrame()
for reg in REG:
    merge1=[]
    REGION1=data[(data.region==reg)]
    #cases
    test1 = pd.merge(REGION1[REGION1.visit=="baseline"],REGION1[REGION1.visit=="followup"],on="id",suffixes=('_baseline', '_followup'))
    #Calculate the differences
    for i in ["betweencentrality","local_eff","cluster_coef"]:
        b=i+"_baseline"
        a=i+"_followup"
        test1[str(i)]=abs(test1[a]-test1[b])
    #integration of genetic data
    #integrate and test
    merge1 = pd.merge(test1,TRY,on="id")
    F=["betweencentrality","local_eff","cluster_coef"]
#    F=[i+"_followup" for i in F]
    G=range(1,len(TRY.columns))
    for f in F:
        reg_p=[]
        for g in G:
            gi=g+15
            X=merge1.iloc[:,gi]
            y=merge1[f]
            X=X.astype(float)
            print(gi,f)
            tryme2=smf.quantreg('y~X',data=merge1).fit(maxiter=100000,q=0.5)
            #print(tryme2.summary())
            #tryme2.pvalues[1]
            #tryme=statsmodels.regression.quantile_regression.QuantReg(y, X.astype(float)).fit()
            #if tryme.pvalues[0] <= 0.000000002:#0.000010288: #0.05÷(90×18×3)
                #count=count+1
            #print (reg,"\n",g,TRY.columns[g],"\n",f,"\n",tryme.pvalues[0])
            reg_p.append(tryme2.pvalues[1] )
        if f=="betweencentrality":
            reg_btw=reg_p
        elif f=="local_eff":
            reg_loc=reg_p
        else:
            reg_clu=reg_p
    col_name=merge1.columns[16:].tolist() 
    ALL=pd.DataFrame({'region':str(reg),'betweeness':reg_btw,'local_eff':reg_loc,'cluster_coef':reg_clu},index=col_name)
    ALL_in=ALL_in.append(ALL)
```


```python
####ALL_in.to_csv("P-values.txt")
```

# Don not run the following unless for CDR???

# Save the data with the differences: NOT ABS


```python
#Here i am trying to create a dataframe of p-values and plot it as heatmap
#RE=REG[:10]
ALL_reg=pd.DataFrame()
for reg in REG:
    merge1=[]
    REGION1=data[(data.region==reg)]
    #cases
    test1 = pd.merge(REGION1[REGION1.visit=="baseline"],REGION1[REGION1.visit=="followup"],on="id",suffixes=('_baseline', '_followup'))
    #Calculate the differences
    for i in ["betweencentrality","local_eff","cluster_coef"]:
        b=i+"_baseline"
        a=i+"_followup"
        test1[str(i)]=test1[b]-test1[a]
    #integration of genetic data
    #integrate and test
    merge1 = pd.merge(test1,TRY,on="id")
    ALL_reg=ALL_reg.append(merge1)
#ALL_reg.to_csv("Differences_All.txt")
```


```python
ALL_reg.head()
```


```python
ALL_reg.shape
```

# Do not run NEXT TWO CELLS (I WAS TESTING)


```python
def get_results(Xx,yy,REGreg,dataa=ALL_reg):
    D=dataa[dataa.region_baseline==REGreg]
    Y=D[yy].astype(float)
    XX=D[Xx].astype(float)
    results=smf.quantreg('Y~XX',data=D).fit(maxiter=100000,q=0.5)
    return {"rsquare":results.prsquared,"pvalue":results.pvalues[1],"t":results.tvalues[1],"beta":results.params[1]}
    #return [results.prsquared,results.pvalues()[1],results.tvalues[1],results.params[1]]
```


```python
# for LaTex purposes

for fe in ["betweencentrality","local_eff","cluster_coef"]:
    with open("new/"+str(fe)+"q.txt","wt") as fout:
        fout.writelines("Gene,Region&Region id&Beta& Statistic&P-value &Local Feature\\\\"+"\n")
        for reg in REG:
            R=ff[ff.region==reg]['AAL_name'].iloc[0]
            for g in new_col[:-1]:
                x=get_results(g, fe ,reg)
                fout.writelines("\\\\bfseries"+"&".join([str(k) for k in [g,R, reg,round(x['beta'],4),round(x['t'],4),round(x['pvalue'],6), fe]])+"\\\\"+"\n")
                #print(g, reg,round(x['beta'],4),round(x['t'],4),round(x['pvalue'],6), f,round(x['rsquare'],4))
    fout.close()
```

# Continue from here: 


```python
def LOG(LIST):
    J=[]
    if type(LIST[0]) is str:
        return(LIST)
    for i in LIST:
        j=-(np.log10(float(i),order=20))
        J.append(j)
    return(J)
```


```python
ALL_final=ALL_in.apply(LOG)
```

# P-Value Distribution


```python
ALL_final.to_csv("new/sign.txt")
```

# Features Association


```python
data_all.corr()
```

# Replotting SS Manhattan


```python
#plot p-values
th=3.2679738562091506e-05

Y= ALL_final.betweeness[ALL_final.betweeness >= -np.log10(float(th))].tolist()[0]
X=int(ALL_final[ALL_final.betweeness >= -np.log10(float(th))].region[0][6:])
label=ALL_final[ALL_final.betweeness >= -np.log10(float(th))].index[0]+": Region"+str(X)
print(X,Y,label)
#plt.rcParams["axes.labelsize"] = 10
sns.set()
sns.set(style="whitegrid", palette="muted",font_scale=1)
# Draw a categorical scatterplot to show each observation
plt.figure(figsize=(15,7 ))
g=sns.swarmplot(x="region", y="betweeness",data=ALL_final,dodge=True,
              palette=["deepskyblue","darkgreen","darkblue","purple"], size=5.5,)#alpha=0.75,)
g.set_xticklabels(s)
plt.xticks(fontsize=15,)# rotation=90)
plt.yticks(fontsize=15 )
plt.ylabel("-log10(pvalue) Betweenness Centrality",fontsize=18)
plt.xlabel("Brain Region",fontsize=18)
plt.annotate(label,
        xy=(X,Y), xytext=(100, -5),
        textcoords='offset pixels', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.1),)#
        #arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.axhline(y=-np.log10(float(th),order=10), color='r', linestyle='-.',linewidth=2.3)
plt.show()


#plot p-values

Y= ALL_final.local_eff[ALL_final.local_eff >= -np.log10(float(th))].tolist()[0]
X=int(ALL_final[ALL_final.local_eff >=-np.log10(float(th))].region[0][6:])
label=ALL_final[ALL_final.local_eff >= -np.log10(float(th))].index[0]+": Region"+str(X)
print(X,Y,label)
sns.set()

sns.set(style="whitegrid", palette="muted",font_scale=1)
# Draw a categorical scatterplot to show each observation
plt.figure(figsize=(15,7 ))
g=sns.swarmplot(x="region", y="local_eff",data=ALL_final,dodge=True,
              palette=["deepskyblue","darkgreen","darkblue","purple"], size=5.5,)#alpha=0.75)
g.set_xticklabels(s)
plt.xticks(fontsize=15,)# rotation=90)
plt.yticks(fontsize=15 )
plt.annotate(label,
        xy=(X,Y), xytext=(100, -5),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.1),)#
        #arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.ylabel("-log10(pvalue) Local Efficiency",fontsize=18)
plt.xlabel("Brain Region",fontsize=18)
plt.axhline(y=-np.log10(float(th),order=10), color='r', linestyle='-.',linewidth=2.3)
plt.show()


#plot p-values
sns.set()

sns.set(style="whitegrid", palette="muted",font_scale=1)
# Draw a categorical scatterplot to show each observation
plt.figure(figsize=(15,7 ))
g=sns.swarmplot(x="region", y="cluster_coef",data=ALL_final,dodge=True,
              palette=["deepskyblue","darkgreen","darkblue","purple"], size=5.5,)#alpha=0.75,)
g.set_xticklabels(s)
plt.xlabel("Brain Region",fontsize=18)
plt.xticks(fontsize=15,)# rotation=90)
plt.yticks(fontsize=15 )
plt.ylabel("-log10(pvalue) Clustering Coefficient",fontsize=18)
plt.axhline(y=-np.log10(float(th),order=10), color='r', linestyle='-.',linewidth=2.3)

```

# To Alex: Plotting the Net Viewer : Do not run


```python
#for v in ['betweeness','local_eff','cluster_coef']:
ALL_final[ALL_final.region==i].T.loc[v,:].index

#ALL_final.pivot(index='region', values='betweencentrality')
'''lc=[v,'region_baseline','id']
cor=data_all[lc]
cor=cor.pivot(index='id', columns='region_baseline', values='betweencentrality')
cor=cor.reindex(['region'+str(i) for i in range(1,91)], axis=1)
cor.columns=[str(i) for i in range(1,91)]
'''

#for i in ['region'+str(i) for i in range(1,91)]:
#to_alex
```


```python
bb={}
cc={}
ll={}
for i in ['region'+str(i) for i in range(1,91)]:
    A=ALL_final[ALL_final.region==i]
    for v in ['betweeness','local_eff','cluster_coef']:
        xi=A[[v]].apply(sum)[0] #['cluter_coeff']
        if v=='betweeness':
            bb[i]=xi
        elif v=='local_eff':
            ll[i]=xi     
        elif  v=='cluster_coef':
            cc[i]=xi
        #print(i,v,xi)
        
```


```python
ff=pd.read_csv("new/alex.txt",header=None,names=['region','reg_id','AAL_name'])
print(ff.head())
bbb=pd.DataFrame(data={'region':list(bb.keys()),'value':list(bb.values())})
print(bbb.head())
ccc=pd.DataFrame(data={'region':list(cc.keys()),'value':list(cc.values())})
print(ccc.head())
lll=pd.DataFrame(data={'region':list(ll.keys()),'value':list(ll.values())})
print(lll.head())
```


```python
lll.merge(ff,on='region').drop(columns=['region','reg_id']).to_csv("new/local_eff.txt")
ccc.merge(ff,on='region').drop(columns=['region','reg_id']).to_csv("new/cluster_coef.txt")
bbb.merge(ff,on='region').drop(columns=['region','reg_id']).to_csv("new/betweeness.txt")
```

# Plotting Spearman

region79 11 BLMH cluster_coef 
 SpearmanrResult(correlation=0.5707881276299926, pvalue=2.797512393067049e-05)
region86 3 APP local_eff 
 SpearmanrResult(correlation=-0.5804810360777058, pvalue=1.8930306634907207e-05)



```python
#sns.set(style="darkgrid")
sns.set(style="whitegrid", palette="muted")#,font_scale=.85)

plot_me= data_all[data_all.region_baseline=='region79']
times1000=[float(i)*1000 for i in plot_me["BLMH"]]
g = sns.jointplot(plot_me.cluster_coef, times1000, kind="reg",color='darkblue')
plt.ylabel("BLMH expression * 1000",fontsize=15)
plt.xlabel("Clustering Coefficient (Heschl_L)",fontsize=15)

#                  xlim=(0, 60), ylim=(0, 12), color="m", height=7)
```


```python
#sns.set(style="darkgrid")
sns.set(style="whitegrid", palette="muted")#,font_scale=.85)

plot_me= data_all[data_all.region_baseline=='region86']
times1000=[float(i)*1000 for i in plot_me["APP"]]
g = sns.jointplot(plot_me.local_eff, times1000, kind="reg",color='darkblue')
plt.ylabel("APP expression * 1000",fontsize=15)
plt.xlabel("Local Efficiency (Temporal_Mid_R)",fontsize=15)

#                  xlim=(0, 60), ylim=(0, 12), color="m", height=7)
```

# Plotting quantiles


```python
#sns.set(style="darkgrid")
sns.set(style="whitegrid", palette="muted")#,font_scale=.85)

plot_me= data_all[data_all.region_baseline=='region55']
times1000=[float(i)*1000 for i in plot_me['PLAU']]
#print(times1000)
print (plot_me['PLAU'].head(n=2))
g = sns.jointplot(times1000,plot_me.betweencentrality, kind="reg",color='darkblue')#hue=data_all.status_baseline)
plt.xlabel("PLAU expression * 1000",fontsize=15)
plt.ylabel("Betweenness Centrality (Region 55)",fontsize=15)

#                  xlim=(0, 60), ylim=(0, 12), color="m", height=7)
```


```python
#sns.set(style="darkgrid")
sns.set(style="whitegrid", palette="muted")#,font_scale=.85)

plot_me= data_all[data_all.region_baseline=='region32']
times1000=[float(i)*1000 for i in plot_me["HFE"]]
#print(times1000)
print(plot_me["HFE"].head(n=2))
g = sns.jointplot(times1000,plot_me.local_eff, kind="reg",color='darkblue')#fit_reg=False)# #hue=data_all.status_baseline)
plt.xlabel("HFE expression * 1000",fontsize=15)
plt.ylabel("Local Efficiency (Region 32)",fontsize=15)

#                  xlim=(0, 60), ylim=(0, 12), color="m", height=7)
```


```python
#ALL_final.sort_values(by='betweeness',ascending=False)
```

# Plotting Quantile line.
## Warning: Want to run? Change data to DATA.


```python
dataa = sm.datasets.engel.load_pandas().data
mod = smf.quantreg('foodexp ~ income', dataa)
res = mod.fit(q=.5)
print(res.summary())



quantiles = np.arange(.05, .96, .1)
def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['income']] + \
            res.conf_int().loc['income'].tolist()
    
models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub'])

ols = smf.ols('foodexp ~ income', dataa).fit()
ols_ci = ols.conf_int().loc['income'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['income'],
           lb = ols_ci[0],
           ub = ols_ci[1])

print(models)
print(ols)

x = np.arange(dataa.income.min(), dataa.income.max(), 50)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle='dotted', color='grey')
    
y = get_y(ols['a'], ols['b'])

ax.plot(x, y, color='red', label='OLS')
ax.scatter(dataa.income, dataa.foodexp, alpha=.2)
ax.set_xlim((240, 3000))
ax.set_ylim((240, 2000))
legend = ax.legend()
ax.set_xlabel('Income', fontsize=16)
ax.set_ylabel('Food expenditure', fontsize=16)
plt.show()

n = models.shape[0]
p1 = plt.plot(models.q, models.b, color='black', label='Quantile Reg.')
p2 = plt.plot(models.q, models.ub, linestyle='dotted', color='black')
p3 = plt.plot(models.q, models.lb, linestyle='dotted', color='black')
p4 = plt.plot(models.q, [ols['b']] * n, color='red', label='OLS')
p5 = plt.plot(models.q, [ols['lb']] * n, linestyle='dotted', color='red')
p6 = plt.plot(models.q, [ols['ub']] * n, linestyle='dotted', color='red')
plt.ylabel(r'$\beta_{income}$')
plt.xlabel('Quantiles of the conditional food expenditure distribution')
plt.legend()
plt.show()
```


```python
sns.set(style="whitegrid", palette="muted")#,font_scale=.85)

plot_me= data_all[data_all.region_baseline=='region55']
times1000=[float(i)*1000 for i in plot_me['PLAU']]
#print(times1000)
print (plot_me['PLAU'].head(n=2))
g = sns.jointplot(times1000,plot_me.betweencentrality, kind="reg",color='darkblue')#hue=data_all.status_baseline)
plt.xlabel("PLAU expression * 1000",fontsize=15)
plt.ylabel("Betweenness Centrality (Region 55)",fontsize=15)
```

# Plot PLAU at region 55 with betweencentrality


```python
#dataa = sm.datasets.engel.load_pandas().data
dataa=data_all[data_all.region_followup=='region55'][['betweencentrality','PLAU']].astype(float)
dataa['PLAU']=dataa.PLAU*1000
mod = smf.quantreg('betweencentrality ~ PLAU', dataa)
res = mod.fit(q=.5)
print(res.summary())
quantiles = [0.5]#np.arange(.05, .96, .1)
def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['PLAU']] + \
            res.conf_int().loc['PLAU'].tolist()
    
models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub'])

ols = smf.ols('betweencentrality ~ PLAU', dataa).fit()
ols_ci = ols.conf_int().loc['PLAU'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['PLAU'],
           lb = ols_ci[0],
           ub = ols_ci[1])
print(models)
print(ols)
x = np.arange(dataa.PLAU.min()-200, dataa.PLAU.max()+200, 50)
#get_y = lambda a, b: a + b * x
sns.set(style="whitegrid", palette="muted")#,font_scale=.85)
plt.figure(figsize=(30,20 ))
#fig, ax = plt.subplots()#
for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    g=(sns.jointplot(dataa.PLAU, dataa.betweencentrality,kind='reg',color='darkblue'))
    plt.plot(x, y,color='red',linewidth=1.5,  linestyle='-',label='OLS' )#.plot_joint(sns.kdeplot, zorder=0, n_levels=6))
    #,hue=data_all.status_baseline)
    y = get_y(ols['a'], ols['b'])
    #plt.plot(x, y, color='black', linewidth=1)
legend = ax.legend()
plt.xlabel("PLAU expression * 1000",fontsize=15)
plt.ylabel("Betweenness Centrality (left fusiform gyrus)",fontsize=15)
############################################################################################################
#dataa = sm.datasets.engel.load_pandas().data
dataa=data_all[data_all.region_followup=='region32'][['local_eff','HFE']].astype(float)
dataa['HFE']=dataa.HFE*1000
mod = smf.quantreg('local_eff ~ HFE', dataa)
res = mod.fit(q=.5)
print(res.summary())
quantiles = [0.5]#np.arange(.05, .96, .1)
def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['HFE']] + \
            res.conf_int().loc['HFE'].tolist()
    
models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub'])

ols = smf.ols('local_eff ~ HFE', dataa).fit()
ols_ci = ols.conf_int().loc['HFE'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['HFE'],
           lb = ols_ci[0],
           ub = ols_ci[1])
print(models)
print(ols)
x = np.arange(dataa.HFE.min()-300, dataa.HFE.max()+330, 50)
#get_y = lambda a, b: a + b * x
sns.set(style="whitegrid", palette="muted")#,font_scale=.85)
plt.figure(figsize=(30,20 ))
#fig, ax = plt.subplots()#
for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    g=(sns.jointplot(dataa.HFE, dataa.local_eff,kind='reg',color='darkblue'))#plot_met_kws={'scatter_kws': {'alpha': 0.1}}))
    plt.plot(x, y,color='red',linewidth=1.5,  linestyle='-',label='OLS' )#.plot_joint(sns.kdeplot, zorder=0, n_levels=6))
    #,hue=data_all.status_baseline)
    y = get_y(ols['a'], ols['b'])
    #plt.plot(x, y, color='black', linewidth=1)
legend = ax.legend()
plt.xlabel("HFE expression * 1000",fontsize=15)
plt.ylabel("Local Efficiency (right anterior cingulate gyrus)",fontsize=15)

```


```python
#dataa = sm.datasets.engel.load_pandas().data
dataa=data_all[data_all.region_followup=='region55'][['betweencentrality','PLAU']].astype(float)
#dataa['PLAU']=dataa.PLAU*1000
mod = smf.quantreg('betweencentrality ~ PLAU', dataa)
res = mod.fit(q=.5)
print(res.summary())



quantiles = np.arange(.05, .96, .1)
def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['PLAU']] + \
            res.conf_int().loc['PLAU'].tolist()
    
models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub'])

ols = smf.ols('betweencentrality ~ PLAU', dataa).fit()
ols_ci = ols.conf_int().loc['PLAU'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['PLAU'],
           lb = ols_ci[0],
           ub = ols_ci[1])

print(models)
print(ols)

x = np.arange(dataa.PLAU.min(), dataa.PLAU.max(), 50)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle='dotted', color='grey')
    
y = get_y(ols['a'], ols['b'])

ax.plot(x, y, color='red', label='OLS')
ax.scatter(dataa.PLAU, dataa.betweencentrality, alpha=.2)
#ax.set_xlim((240, 3000))
#ax.set_ylim((240, 2000))
legend = ax.legend()
ax.set_xlabel('PLAU', fontsize=16)
ax.set_ylabel('Betweenness Centrality', fontsize=16)
plt.show()

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

fig, ax = plt.subplots()
n = models.shape[0]
p1 = plt.plot(models.q, models.b, color='red', label='Quantile Reg.')
p2 = plt.plot(models.q, models.ub, linestyle='dotted', color='red')
p3 = plt.plot(models.q, models.lb, linestyle='dotted', color='red')
p4 = plt.plot(models.q, [ols['b']] * n, color='darkblue', label='OLS')
p5 = plt.plot(models.q, [ols['lb']] * n, linestyle='dotted', color='darkblue')
p6 = plt.plot(models.q, [ols['ub']] * n, linestyle='dotted', color='darkblue')
plt.ylabel(r'$\beta_{PLAU}$ at left fusiform gyrus',fontsize=15)
plt.xlabel('Quantiles of the conditional betweenness centrality distribution',fontsize=15)
plt.legend()
plt.show()
```

# Plot HFE at 32 with local efficiency


```python
#dataa = sm.datasets.engel.load_pandas().data
dataa=data_all[data_all.region_followup=='region32'][['local_eff','HFE']].astype(float)
#dataa['HFE']=dataa.HFE*1000
mod = smf.quantreg('local_eff ~ HFE', dataa)
res = mod.fit(q=.5)
print(res.summary())



quantiles = np.arange(.05, .96, .1)
def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['HFE']] + \
            res.conf_int().loc['HFE'].tolist()
    
models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub'])

ols = smf.ols('local_eff ~ HFE', dataa).fit()
ols_ci = ols.conf_int().loc['HFE'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['HFE'],
           lb = ols_ci[0],
           ub = ols_ci[1])

print(models)
print(ols)

x = np.arange(dataa.HFE.min(), dataa.HFE.max(), 50)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle='dotted', color='grey')
    
y = get_y(ols['a'], ols['b'])

ax.plot(x, y, color='red', label='OLS')
ax.scatter(dataa.HFE, dataa.local_eff, alpha=.2)
#ax.set_xlim((240, 3000))
#ax.set_ylim((240, 2000))
legend = ax.legend()
ax.set_xlabel('HFE', fontsize=16)
ax.set_ylabel('Local Efficiency', fontsize=16)
plt.show()
sns.set()
sns.set(rc={'axes.facecolor':'white' ,'figure.facecolor':'white'})

n = models.shape[0]
p1 = plt.plot(models.q, models.b, color='red', label='Quantile Reg.')
p2 = plt.plot(models.q, models.ub, linestyle='dotted', color='red')
p3 = plt.plot(models.q, models.lb, linestyle='dotted', color='red')
p4 = plt.plot(models.q, [ols['b']] * n, color='darkblue', label='OLS')
p5 = plt.plot(models.q, [ols['lb']] * n, linestyle='dotted', color='darkblue')
p6 = plt.plot(models.q, [ols['ub']] * n, linestyle='dotted', color='darkblue')
plt.ylabel(r'$\beta_{HFE}$ at right anterior cingulate gyrus',fontsize=15)
plt.xlabel('Quantiles of the conditional local efficiency distribution',fontsize=15)
plt.legend()
plt.show()
```

# Global Features

## Integrating the Datasets


```python
GLOBAL=pd.read_csv("../Local_Features_Differences.csv")
GLOBAL.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True)
GLOBAL.columns
GLOBAL.columns=['id', 'louvain_baseline', 'char_path_len_baseline', 'global_eff_baseline',
       'transitivity_baseline', 'Visit_baseline', 'louvain_followup', 'char_path_len_followup',
       'global_eff_followup', 'transitivity_followup', 'Visit_y', 'Status',
       'transitivity', 'global_eff', 'louvain', 'char_path_len']
####GLOBAL.to_csv("Global.csv")
```

# Edit: the difference without absolute value


```python
GLOBAL.columns
```


```python
def edit_sign(d):
    cols=['transitivity', 'global_eff', 'louvain', 'char_path_len']
    for i in cols:
        bs=i+'_baseline'
        fl=i+'_followup'
        h=str(i)+"_diff"
        d[i]=d[[fl,bs,i]].apply(lambda x: -1*x[i] if x[fl] < x[bs] else x[i], axis=1)
    return d
```


```python
GLOBAL_final=edit_sign(GLOBAL)
```


```python
GLOBAL_final.to_csv("new/Global_final.csv")
```

# Importing CDR dataset`


```python
CDR=pd.read_csv("CDR.csv")
```


```python
CDR.columns
```

# Adding RID to GLOBAL dadaset


```python
def intrid(c):
    '''intrid is used to define an RID column in the GLOBAL data using the available id column'''
    return int(c.split("_")[2])
GLOBAL_final['RID']=GLOBAL_final["id"].apply(intrid)
```


```python
#sc m12 
#Here I am calculating a CDR-baseline and CDR_followup for all patients - redundant in each patient.
def cdr_visits(d):
    '''It takes the data frame for a certain id, and calculates the baseline and follow-up score for each one, and the sums'''
    d1 = pd.merge(d[d.VISCODE2=="sc"],d[d.VISCODE2=="m12"],on="RID",suffixes=('_baseline', '_followup'))
    return d1
    
def cdr_diff(d):
    '''This function is to calculate the cdr individual differences, sums in both visits, and diff and abs diff of the sum'''
    cdrs=['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE','CDGLOBAL']
    for i in cdrs:
        b=i+'_baseline'
        f=i+'_followup'
        d[str(i)]=d[b]-d[f]
    bs=[k+'_baseline' for k in ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE','CDGLOBAL']]
    fl=[k+'_followup' for k in ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE','CDGLOBAL']]
    d['CDR_baseline']=d[bs].sum(axis=1)
    d['CDR_followup']=d[fl].sum(axis=1)
    d['CDR_diff']=d['CDR_baseline']-d['CDR_followup']
    return d

```


```python

CDR_Visits=cdr_visits(pd.read_csv("CDR.csv"))
```


```python
CDR_final = cdr_diff(cdr_visits(CDR))
```


```python
CDR_final.to_csv('new/CDR_Differences.csv')
```


```python
CDR_final.columns
```

# Importing Gene data: Imported Already Above

# Adding RID to gene data


```python
TRY['RID']=TRY["id"].apply(intrid)
```

# Merge data 


```python
#integration of genetic data 
merge = pd.merge(GLOBAL_final,TRY,on="RID")
#Integration of CDR data
merge1= pd.merge(merge,CDR_final,on="RID")
```


```python
merge1.columns
```


```python
merge1.to_csv("new/Chapter4.csv")
```


```python
merge1.to_csv("Chapter4.csv")
```


```python
merge1.to_csv("new/Chapter4_copy.csv")
```


```python
#Define the genes, features and CDRs (differences)

F=['transitivity', 'global_eff', 'louvain', 'char_path_len']
G=merge1.loc[:,'APBB2':'ABCA7'].columns
C=merge1.loc[:,['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN','CDHOME', 'CDCARE','CDGLOBAL', 'CDR_diff']].columns
```

# Plot baseline and follow-up for global features


```python
pl=['transitivity','global_eff', 'louvain', 'char_path_len']
PL={'transitivity':'Transitivity','global_eff':'Global Effect', 'louvain':'Louvain Modularity', 'char_path_len':'Characteristic Path Length'}
s1='_baseline'
s2='_followup'

for j in pl:
    x={}
    y={}
    x[str(j)]=merge1[j+s1]
    x["Status"]=merge1['Status']
    y[str(j)]=merge1[j+s2]
    y["Status"]=merge1['Status']
    x['Visit']=['Baseline' for c in range(len(x[str(j)]))]
    y['Visit']=['Follow-up' for c in range(len(y[str(j)]))]
    D=pd.DataFrame(data=x)
    D=D.append(pd.DataFrame(data=y))
    sns.set(style="ticks", palette="pastel",font_scale=1.4)

    # Load the example tips dataset
    tips = sns.load_dataset("tips")

    # Draw a nested boxplot to show bills by day and time
    Boxplot=sns.boxplot(x="Visit", y=j,
                hue="Status", palette=["blue", "gold"],
                data=D)#,legend=False)
    #Boxplot.despine(left=True)
    plt.legend(loc='upper left',labelspacing=0.0,borderpad=0.0)
    plt.setp(Boxplot.get_legend().get_texts(), fontsize='10') # for legend text
    plt.ylabel(PL[j])
    sns.despine( trim=True)
    plt.show()
    
    #print(merge1[x])
```

# Plotting the CDR


```python
pl=['CDMEMORY', 'CDORIENT', 'CDJUDGE',
       'CDCOMMUN', 'CDHOME', 'CDCARE',"CDGLOBAL"]
s1='_baseline'
s2='_followup'

for j in pl:
    x={}
    y={}
    x[str(j)]=merge1[j+s1]
    x["Status"]=merge1['Status']
    x["RID"]=merge1['RID']
    y[str(j)]=merge1[j+s2]
    y["Status"]=merge1['Status']
    y["RID"]=merge1['RID']
    x['Visit']=['Baseline' for c in range(len(x[str(j)]))]
    y['Visit']=['Follow-up' for c in range(len(y[str(j)]))]
    D=pd.DataFrame(data=x)
    D=D.append(pd.DataFrame(data=y))
    sns.set()
    sns.set(style="whitegrid", palette="muted",font_scale=1.4,rc={'axes.facecolor':'white' ,'figure.facecolor':'white'})
    # Draw a nested boxplot to show bills by day and time
    #Boxplot=sns.lineplot(x='Visit', y=j,alpha=0.35,dashes=True, units="RID",markers=True,estimator=None,hue="Status",style="Status",palette="ch:2.5,6.9",hue_order=["AD","CONTROL"], data=D,)#palette=["r","c"], data=D,size=8,)#orient='h',alpha=0.8,)#kind='point')
#    for index, row in D.iterrows():
#        Boxplot.text(row.Visit,row[j],'test')# round(row[j],2),ha="right")#, color='black', ha="left")
    Boxplot=sns.violinplot(data=D,y=j,x='Visit',inner=None,palette=['gold','gold'],alpha=0.3)#hue=0.3,color="gold")#,"olivedrab"])
    Boxplot=sns.swarmplot(x='Visit', y=j, hue="Status",alpha=0.7,hue_order=["AD","CONTROL"],palette=["red","darkblue"], data=D,size=8,orient='v')#orient='h')#orient='h')#orient='h',alpha=0.8,)#kind='point')
    #swarmplot
    #Boxplot.despine(left=True)
    plt.legend(loc='upper center',labelspacing=0.0,borderpad=0.0)
    plt.setp(Boxplot.get_legend().get_texts(), fontsize='11') # for legend text
    #sns.despine(offset=10, trim=True)
    plt.show()

```

# Do not run this:


```python
#Test the QR - don not run

X=merge1[['char_path_len','ABCA7']].astype(float)
Y= merge1['CDHOME'].astype(float)
tryme2=smf.quantreg('Y~X',data=merge1).fit(maxiter=100000,q=0.5)
tryme2.pvalues
tryme2.params
tryme2.summary()
tryme2.params[2]
```

# Quantile Regressio 


```python
#Define a functio to perform quantile regression on each gene (OR, cdr), vs feature
def QR(f,g,df=merge1):
    reg_p=[]
    Y=df[f].astype(float)
    for i in g:
        X=df[i].astype(float)
        tryme2=smf.quantreg('Y~X',data=df).fit(maxiter=100000,q=0.5)
        reg_p.append(tryme2.pvalues[1] )
    return reg_p
```


```python
def QR_beta(f,g,df=merge1):
    reg_p=[]
    Y=df[f].astype(float)
    for i in g:
        X=df[i].astype(float)
        tryme2=smf.quantreg('Y~X',data=df).fit(maxiter=100000,q=0.5)
        reg_p.append(str(round(tryme2.params[1],8))+" ("+str(round(tryme2.pvalues[1],4))+" )" )#reg_p.append(tryme2.pvalues[1] )
    return reg_p
```


```python
print(0.05/(len(G)*len(F)))
print(0.05/(len(C)-1))
print(len(C)-1)
```


```python
#Save p-values
###gene_quantile.to_csv("Quantile_Global_gene.csv")
```


```python
####ALL_global_final.to_csv("Global_quantile_cdr.txt")
```


```python
#just to check previous results
```

# Quantile with CDR as Y, MRI as X

## Here we go!


```python
#Create dectionaty for a dataframe of p-values
D={}
D['Global Features']=F
for i in C:
    D[i]=QR_beta(i,g=F)

#Create a dataframe with all p-values
globalFeatures_cdr_quantile=pd.DataFrame(data=D)
#Save p-values
#globalFeatures_cdr_quantile.to_csv("new/globalFeatures_cdr_quantile.csv",sep="&")
print(globalFeatures_cdr_quantile)
```

# Ridge of CDR on BOTH imaging and genes


```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
```

# Functions: Select alpha, and conduct ridge based on alpha


```python
ALPHA=list(np.arange(1e-20,1,1e-1))+list(np.arange(1,20,1))
```


```python
ALPHA=list(np.arange(1e-20,1,1e-1))+list(np.arange(1,20,1))

def select_alpha(x,y):
    '''Select the best alpha for the model'''
    ridge=Ridge()#####Lasso()
    parameters={'alpha':ALPHA}#[1e-15,1e-10,1e-8,1e-4,1e-3,1e-3,1,5,10,20]}
    ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
    ridge_regressor.fit(x,y)
    return(ridge_regressor.best_params_['alpha'], ridge_regressor.best_score_)

def fit_ridge(C,f,x2,alpha,df=merge1):
    '''To fit Ridge. C is the y (CDR), X is the featuers (genes and global/local features)'''
    Y=df[C].astype(float) #CDR
    cos=(" ".join([f])+" "+" ".join(list(x2))).split()
    X=df[cos].astype(float)
    clf = Ridge(alpha=alpha,normalize=True,fit_intercept=False)#### Lasso(alpha=alpha,normalize=True)
    return clf.fit(X, Y)

def fit_ridge_plot(C,f,x2,alpha,df=merge1):
    '''To fit Ridge. C is the y (CDR), X is the featuers (genes and global/local features)'''
    Y=df[C].astype(float) #CDR
    cos=(" ".join([f])+" "+" ".join([x2])).split()
    X=df[cos].astype(float)
    clf = Ridge(alpha=alpha,normalize=True,fit_intercept=False)#### Lasso(alpha=alpha,normalize=True)
    return clf.fit(X, Y)
    #reg_p.append(str(round(tryme2.params[1],8))+" ("+str(round(tryme2.pvalues[1],4))+" )" )#reg_p.append(tryme2.pvalues[1] )
    #return reg_p
```

# Defining X as the Imaging differences AND the gene expressions - and Y as the CDR


```python
ALPHA=list(np.arange(1e-20,1,1e-1))+list(np.arange(1,150,1))
fout=open("new/ridge.txt","wt")
fout.writelines("&".join(['CDR','Global Metric','Alpha','score',]+list(G))+"\n")
for i in C:
    for f in F:
        Y=merge1[i].astype(float)
        cos=(" ".join([f])+" "+" ".join(list(F))).split()
        X=merge1[cos].astype(float)
        alp,score=select_alpha(X,merge1[i])
        sol=fit_ridge(i,f,G,alpha=alp)
        fout.writelines("&".join([i,g,str(alp),str(score)]+[str(i) for i in sol.coef_])+"\\\\"+"\n")
```


```python
fout.close()
```

# Plot Ridge


```python
print(np.arange(1e-20,1,2e-1))
```


```python
ALPHA=list([.05,0.1,0.2,0.5,0.7,0.99]+list(np.arange(1,250,10)))

for i in C:
    for f in F:
        coefs = []
        for alp in ALPHA:
            Y=merge1[i].astype(float)
            cos=(" ".join([f])+" "+" ".join(list(G))).split()
            X=merge1[cos].astype(float)
            sol=fit_ridge(i,f,G,alpha=alp)
            #print (i,G,alp,score,sol.coef_)
            coefs.append(sol.coef_)
        '''        
        for alp in ALPHA:
            for g in G:
                Y=merge1[i].astype(float)
                cos=(" ".join([f])+" "+" ".join([g])).split()
                X=merge1[cos].astype(float)
                sol=fit_ridge(i,f,g,alpha=alp,)#fit_intercept=False) > take this to the function
                print (i,alp,score,sol.coef_)
                coefs.append(sol.coef_)'''

        # #############################################################################
        # Display results

        ax = plt.gca()

        ax.plot(ALPHA, coefs)
        ax.set_xscale('log')
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Ridge coefficients as a function of the regularization: '+i+" on "+f)
        plt.axis('tight')
        plt.show()
```

# LASSO?


```python
from sklearn.linear_model import Lasso
lasso=Lasso(normalize=True)
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-3,1,5,10,20]}
lasso_regressor=GridSearchCV(lasso, parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,Y['CDR_diff'],)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

```

# Spearman Associations


```python
def SP(f,g,df=merge1):
    reg_p=[]
    Y=df[f].astype(float)
    for i in g:
        X=df[i].astype(float)
        tryme2=stats.spearmanr(Y,X)
        reg_p.append(str(round(tryme2[0],4))+" ("+str(round(tryme2[1],4))+" )" )
    return reg_p

def SP_p(f,g,df=merge1):
    reg_p=[]
    Y=df[f].astype(float)
    for i in g:
        X=df[i].astype(float)
        tryme2=stats.spearmanr(Y,X)
        reg_p.append(round(tryme2[1],4) )
    return reg_p
```


```python
print(len(C),len(F))
print(len(C)*len(F))
print(0.05/(len(C)-1))

```


```python
#Create dectionaty for a dataframe of p-values
D={}
D['CDR']=C
for i in F:
    D[i]=SP(i,g=C)

#Create a dataframe with all p-values
cdr_sp_quantile=pd.DataFrame(data=D)
#Save p-values
cdr_sp_quantile.T.to_csv("new/cdr_sp_quantile.csv",sep="&")
cdr_sp_quantile.T
```

# NOTE: Change to the absolute change: Done that in the following function

# Global change with gene expressions


```python
#Edit this like before

def get_results_global(Xx,yy,dataa=merge):
    Y=abs(dataa[yy].astype(float))
    XX=dataa[Xx].astype(float)
    results=smf.quantreg('Y~XX',data=dataa).fit(maxiter=100000,q=0.5)
    return {"rsquare":results.prsquared,"pvalue":results.pvalues[1],"t":results.tvalues[1],"beta":results.params[1]}
    #return [results.prsquared,results.pvalues()[1],results.tvalues[1],results.params[1]]
```


```python
for f in 'transitivity,global_eff,louvain,char_path_len'.split(','):
    fout=open("new/"+f+"_q_g.txt","wt")
    fout.writelines("Gene&Beta& Statistic&P-value &Metric\\\\"+"\n")
    for g in G:
        #f='local_eff'
        x=get_results_global(g, f ,)
        fout.writelines("&".join([str(k) for k in [g,round(x['beta'],4),round(x['t'],4),round(x['pvalue'],6), f]])+"\\\\"+"\n")
        #print(g,round(x['beta'],4),round(x['t'],4),round(x['pvalue'],6), f)
        #print(get_results('HFE','local_eff','region32'))
#print(tryme2.prsquared,tryme2.pvalues[1],tryme2.tvalues[1],tryme2.params[1])
fout.close()
```

# Spearman for globals


```python
def spearman_global(Xx,yy,dataa=merge):
    Y=abs(dataa[yy].astype(float))
    XX=dataa[Xx].astype(float)
    results=stats.spearmanr(Y,XX)
    return {"pvalue":results[1],"r":results[0]}
    #return [results.prsquared,results.pvalues()[1],results.tvalues[1],results.params[1]]
```


```python
for f in 'transitivity,global_eff,louvain,char_path_len'.split(','):
    fout=open("new/"+f+"_sp_g.txt","wt")
    fout.writelines("Gene&Beta& r&P-value &Metric\\\\"+"\n")
    for g in G:
        #f='local_eff'
        x=spearman_global(g, f ,)
        fout.writelines("&".join([str(k) for k in [g,round(x['r'],4),round(x['pvalue'],4), f]])+"\\\\"+"\n")
        #print(g,round(x['r'],4),round(x['pvalue'],4), f)
        #print(get_results('HFE','local_eff','region32'))
#print(tryme2.prsquared,tryme2.pvalues[1],tryme2.tvalues[1],tryme2.params[1])
fout.close()
```


```python
print(merge.shape)
print(merge1.shape)
```


```python
round(0.05/17,4)

```

# CDR on Images for the Local features

# Integrate all


```python
ALL_reg.head()
```

# ADD RID to ALL_reg: to the data with all differences (NO ABS)


```python
def intrid(c):
    '''intrid is used to define an RID column in the GLOBAL data using the available id column'''
    return int(c.split("_")[2])
ALL_reg['RID']=ALL_reg["id"].apply(intrid)
```

# Merge the ALL_reg with CDR


```python
ALL_cdr= pd.merge(ALL_reg,CDR_final,on="RID")
```


```python
print(ALL_cdr.shape)
print(ALL_reg.shape)
```


```python
def QR_beta_local(f,g,reg,df=ALL_cdr):
    reg_p={}
    df=df[df.region_followup==reg]
    Y=df[f].astype(float)
    i=g
    X=df[i].astype(float)
    tryme2=smf.quantreg('Y~X',data=df).fit(maxiter=100000,q=0.5)
    reg_p['pvalue']=round(tryme2.pvalues[1],70)
    reg_p['beta']=round(tryme2.params[1],10)
    return reg_p
```


```python
FL= ['local_eff' ,'cluster_coef', 'betweencentrality']
```


```python
# Create dectionaty for a dataframe of p-values
fi=open("new/cdr_on_local.txt","wt")
fi.writelines('&'.join('CDR,Metric,Region,Region id,$\beta$,P-value'.split(','))+"\\\\"+"\n")
for f in FL:
    for l in L:
        r='region'+str(l)
        R=ff[ff.region==r]['AAL_name'].iloc[0]
        for i in C:
            result=QR_beta_local(i,g=f,reg=r)
            fi.writelines("\\bfseries  "+" & ".join([str(k) for k in [i,f,R,r,result['beta'],result['pvalue']]])+"\\\\\n")
```


```python
1894/90
```


```python
0.05/(6*90)

```


```python
124-53

```


```python

```

