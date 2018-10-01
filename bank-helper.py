import pandas as pd

bank = pd.read_csv('bank.csv',sep=';')

bank.to_csv('bank-modified.csv')