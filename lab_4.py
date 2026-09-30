import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('ml-start/data/titanic.csv')
print(df.shape); print(df.head()); df.info(); print(df.describe())

kids = df[df['age'] < 12]
print(len(kids), kids['survived'].mean())

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])
print(df.isna().sum())

print(df.groupby('sex')['survived'].mean())
print(df.groupby('pclass')['survived'].mean())
print(df.groupby(['pclass', 'sex'])['age'].mean().round(1))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].hist(df['age'], bins=30); ax[0].set_title('Возраст')
sns.boxplot(x='pclass', y='fare', data=df, ax=ax[1]); ax[1].set_title('Цена по классу')
sns.barplot(x='sex', y='survived', data=df, ax=ax[2]); ax[2].set_title('Выживаемость')
fig.tight_layout(); fig.savefig('titanic.png', dpi=120)
plt.show()

df['family'] = df['sibsp'] + df['parch']
df['sex_num'] = df['sex'].map({'female': 1, 'male': 0})
df.to_csv('ml-start/data/clean.csv', index=False)

# Создаем новое окно для графика
plt.figure(figsize=(8, 5))

# Фильтруем данные: отдельно для выживших и погибших
survived = df[df['survived'] == 1]['fare']
died = df[df['survived'] == 0]['fare']

# Строим две гистограммы на одном графике с прозрачностью alpha=0.5
plt.hist(died, bins=40, alpha=0.5, label='Погибли', color='red')
plt.hist(survived, bins=40, alpha=0.5, label='Выжили', color='green')

# Добавляем оформление
plt.title('Распределение стоимости билетов для выживших и погибших')
plt.xlabel('Цена билета (Fare)')
plt.ylabel('Количество пассажиров')
plt.legend()  # Показывает окошко с пояснением цветов (выжили/погибли)

# Ограничим ось X до 200-300, чтобы из-за редких сверхдорогих билетов график не сжимался
plt.xlim(0, 250) 

# Сохраняем и показываем
plt.savefig('fare.png', dpi=120)
plt.show()
