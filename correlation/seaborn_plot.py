import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
  
tips = pd.read_csv('NC_metrics.csv')
ax = sns.catplot(x="Metric", y="Correlation",  kind="box", legend=True, hue="Metric", data=tips, palette="Blues",legend_out=False,width=0.75, dodge=False)  
 
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()
 
