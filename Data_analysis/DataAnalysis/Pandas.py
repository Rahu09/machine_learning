import pandas as pd

XYZ_web = {'Day':[1,2,3,4,5,6], 'Visitors':[1000,700,6000,1000,400,350], 'Bounce_Rate':[20,20,23,15,10,34]}

df = pd.DataFrame(XYZ_web)

# showing the dataframe
print(df)

# slicing the dataframe
print("\nsliced verson of dataframe is\n")
print(df.head(2)) #to show first two coulmns as a slice
print(df.tail(2)) #to show the last two column as a slice