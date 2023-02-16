import pandas as pd

country = pd.read_csv('name.csv',index_col=0)
country.to_html('new_name.html')