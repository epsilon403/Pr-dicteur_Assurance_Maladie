import pandas as pd

df = pd.read_csv('assurance-maladie-68d92978e362f464596651.csv')
# print(df.info())
# print(df.describe())
# print(df.head())

ax = df['age'].plot(kind='hist' , bins= 20)
ax = df['BIM'].plot(kind='hist' , bins= 20)
ax = df['age'].plot(kind='hist' , bins= 20)














# def main():
#     print("Hello from pr-dicteur-assurance-maladie!")


# if __name__ == "__main__":
#     main()
