import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import style
from numpy.random import randn
style.use("fivethirtyeight")
np.random.seed(101)

XYZ_web = {'Day':[1,2,3,4,5,6], 'Visitors':[1000,700,6000,1000,400,350], 'Bounce_Rate':[20,20,23,15,10,34]}

df = pd.DataFrame(XYZ_web)

# showing the dataframe
# print(df)

# slicing the dataframe
# print("\nsliced verson of dataframe is\n")
# print(df.head(2)) #to show first two coulmns as a slice
# print(df.tail(2)) #to show the last two column as a slice


# assigning names to rows and coulmn
df0 = pd.DataFrame(randn(3,3),['A','B','C'],['D','E','F'])
# print(df0['E'])

# to drop a row
print(df0.drop('C'))

#two data frames(tables)
df1 = pd.DataFrame({"HPI":[80,90,70,60], "INT_RATE":[2,1,2,3], "IND_GDP":[50,45,45,67]}, index= [2001, 2002,2003,2004])
df2 = pd.DataFrame({"HPI":[80,90,70,60], "INT_RATE":[2,1,2,3], "IND_GDP":[50,45,45,67]}, index= [2005, 2006,2007,2008])

# merging them
merge = pd.merge(df1, df2)
# print(merge)

#only merging given column
merge1 = pd.merge(df1,df2, on = "HPI" and "INT_RATE")
# print(merge1)

#two data frames(tables)
df3 = pd.DataFrame({"INT_RATE":[2,1,2,3], "IND_GDP":[50,45,45,67]}, index= [2001,2002,2003,2004])
df4 = pd.DataFrame({"Low_Tier_HPI":[60,45,67,34], "Unemployment":[1,3,5,6]}, index= [2001,2003,2004,2004])

# joining them
joined = df3.join(df4)
# print(joined)

# ploting a graph using matplotlib
df5 = pd.DataFrame({"Day":[1,2,3,4], "visitors":[200,100,230,300],"Bounce_Rate":[20,45,60,10]})
# df5.plot()
# plt.show()

# changing coloumn header
df5 = df5.rename(columns={"visitors":"Users"})
# print(df5)

# concatination of two dataframes.
concat = pd.concat([df1,df2])
# print(concat)
