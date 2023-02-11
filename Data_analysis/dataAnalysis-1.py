import pandas as pd
import matplotlib as plt
from matplotlib import style
style.use("fivethirtyerght")



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
print(joined)

# 