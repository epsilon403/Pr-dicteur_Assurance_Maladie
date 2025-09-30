import pandas as pd
import seaborn as sns
import matplotlib as plt


df = pd.read_csv('assurance-maladie-68d92978e362f464596651.csv')
print(df.info())
print(df.describe())
print(df.head())
import seaborn as sns
import matplotlib as plt

sns.boxplot(x = 'smoker' , y = 'bmi' , data = df)
plt.title("Seaborn Box Plot by Category")
plt.show()