import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

subjects = pd.read_csv('subject_user.csv')
consumption = pd.read_csv('consumption_user.csv')

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# 2. ANALIZA DESCRIPTIVA: PERSOANE

print("--- Analiză Demografică ---")

# A. Distributia Varstei
plt.figure()
sns.histplot(subjects['AGE_YEAR'], bins=20, kde=True, color='skyblue')
plt.title('Distribuția Vârstei Subiecților (România)', fontsize=14)
plt.xlabel('Vârstă (Ani)')
plt.ylabel('Număr Persoane')
plt.savefig('age_distribution.png')

# B. Distributia pe Sexe (1=Masculin, 2=Feminin)
plt.figure(figsize=(6, 5))
subjects['SEX'].value_counts().plot(kind='bar', color=['salmon', 'lightgreen'])
plt.title('Distribuția pe Gende (1=M, 2=F)')
plt.xticks(ticks=[0, 1], labels=['Feminin', 'Masculin'], rotation=0)
plt.ylabel('Număr Persoane')
plt.savefig('gender_distribution.png')

print(f"Total subiecți analizați: {len(subjects)}")
print(f"Vârsta medie: {subjects['AGE_YEAR'].mean():.2f} ani\n")


# 3. ANALIZA DESCRIPTIVA: CONSUM ALIMENTAR

print("--- Analiză Consum Alimentar ---")

# Analiza Esantionului 
total_subiecti = len(subjects)
nr_judete = subjects['ADM1_NAME'].nunique()
top_judete = subjects['ADM1_NAME'].value_counts().head(3)

print("--- ANALIZA EȘANTIONULUI ---")
print(f"Număr total de persoane: {total_subiecti}")
print(f"Număr de județe reprezentate: {nr_judete}")
print("Cele mai reprezentate județe:")
print(top_judete)
print("-" * 30)

# Diversitatea Dietei
ingr_totale_zi = consumption.groupby(['SUBJECT', 'SURVEY_DAY'])['INGREDIENT_ENG'].nunique().mean()

# Filtrarea ingredientelor omniprezente (Sare, Ulei, Apă, Zahar, Faina, Piper)
exclude_list = ['SALT', 'WATER (DRINKING  WATER)', 'OIL', 'SUGAR', 'FLOUR', 'WATER', 'PEPPER']
filtered_cons = consumption[~consumption['INGREDIENT_ENG'].isin(exclude_list)]

# Calculam media dupa filtrare
ingr_filtrate_zi = filtered_cons.groupby(['SUBJECT', 'SURVEY_DAY'])['INGREDIENT_ENG'].nunique().mean()

print("--- DIVERSITATEA DIETEI ---")
print(f"Media ingrediente/zi (total): {ingr_totale_zi:.2f}")
print(f"Media ingrediente/zi (după filtrare): {ingr_filtrate_zi:.2f}")
print("-" * 30)

# Grupe Alimentare Dominante
dominant_items = consumption['INGREDIENT'].value_counts().head(10)
print("--- TOP 10 INGREDIENTE DOMINANTE ---")
print(dominant_items)
print("-" * 30)

# Distribuția pe Genuri (1=Masculin, 2=Feminin)
# Unim datele de consum cu datele demografice
data_merged = consumption.merge(subjects[['SUBJECT', 'SEX']], on='SUBJECT')

# Definim cuvinte cheie pentru categorii
def count_category(df, keywords):
    return df['INGREDIENT_ENG'].str.contains('|'.join(keywords), case=False, na=False)

fruit_kw = ['FRUIT', 'APPLE', 'BERRY', 'CITRUS', 'PEACH', 'GRAPE']
veg_kw = ['VEGETABLE', 'TOMATO', 'ONION', 'CARROT', 'POTATO', 'CABBAGE', 'SALAD']
meat_kw = ['MEAT', 'PORK', 'CHICKEN', 'BEEF', 'POULTRY', 'SAUSAGE', 'HAM']
alc_kw = ['WINE', 'BEER', 'SPIRIT', 'ALCOHOL', 'LIQUOR', 'VODKA']

data_merged['is_fruit'] = count_category(data_merged, fruit_kw)
data_merged['is_veg'] = count_category(data_merged, veg_kw)
data_merged['is_meat'] = count_category(data_merged, meat_kw)
data_merged['is_alcohol'] = count_category(data_merged, alc_kw)

# Calculam media aparitiilor per zi per gen
gender_stats = data_merged.groupby(['SEX', 'SUBJECT', 'SURVEY_DAY']).agg({
    'is_fruit': 'sum',
    'is_veg': 'sum',
    'is_meat': 'sum',
    'is_alcohol': 'sum'
}).reset_index().groupby('SEX').mean()

print("--- TENDINȚE PE GENURI (Medii per zi) ---")
print("Cod SEX: 1 = Masculin, 2 = Feminin")
print(gender_stats[['is_fruit', 'is_veg', 'is_meat', 'is_alcohol']])

# A. Top 15 Ingrediente
top_15_ro = consumption['INGREDIENT'].value_counts().head(15)
plt.figure()
sns.barplot(x=top_15_ro.index, y=top_15_ro.values, palette='viridis')
plt.title('Top 15 cele mai frecvente ingrediente (România)', fontsize=15)
plt.ylabel('Frecvență (Număr apariții)')
plt.xlabel('Ingredient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_15_ingrediente_ro.png')

# B. Media de ingrediente consumate pe zi
ingr_per_day = consumption.groupby(['SUBJECT', 'SURVEY_DAY'])['INGREDIENT'].nunique()
print(f"Media de ingrediente distincte per persoană/zi: {ingr_per_day.mean():.2f}")

# 4. DESCOPERIREA REGULILOR DE ASOCIERE 
print("\n--- Generare Reguli de Asociere ---")

# Filtrare ingrediente generice
exclude_list_ro = ['SARE', 'APA', 'ULEI', 'ZAHAR', 'FAINA', 'PIPER']
filtered_consumption = consumption[~consumption['INGREDIENT'].isin(exclude_list_ro)]

# Crearea tranzactiilor (O zi de consum a unui individ)
transactions = filtered_consumption.groupby(['SUBJECT', 'SURVEY_DAY'])['INGREDIENT'].apply(set).tolist()
num_transactions = len(transactions)

# Identificarea itemilor frecventi (Suport minim 5%)
item_counts = {}
for t in transactions:
    for item in t:
        item_counts[item] = item_counts.get(item, 0) + 1

frequent_items = {item: count / num_transactions for item, count in item_counts.items() 
                  if count / num_transactions >= 0.05}

# Identificarea perechilor frecvente
pair_counts = {}
for t in transactions:
    items_in_t = [i for i in t if i in frequent_items]
    items_in_t.sort()
    for i in range(len(items_in_t)):
        for j in range(i + 1, len(items_in_t)):
            pair = (items_in_t[i], items_in_t[j])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

# Calcularea metricilor: Suport, Încredere, Lift
rules_list = []
for (item_a, item_b), count in pair_counts.items():
    support_ab = count / num_transactions
    if support_ab >= 0.05:
        # Regula A -> B
        conf_a_b = support_ab / frequent_items[item_a]
        lift = conf_a_b / frequent_items[item_b]
        rules_list.append({
            'Antecedent': item_a, 
            'Consecvent': item_b, 
            'Suport': round(support_ab, 4), 
            'Incredere': round(conf_a_b, 4), 
            'Lift': round(lift, 2)
        })
        # Regula B -> A
        conf_b_a = support_ab / frequent_items[item_b]
        lift_ba = conf_b_a / frequent_items[item_a]
        rules_list.append({
            'Antecedent': item_b, 
            'Consecvent': item_a, 
            'Suport': round(support_ab, 4), 
            'Incredere': round(conf_b_a, 4), 
            'Lift': round(lift_ba, 2)
        })

# Crearea si salvarea tabelului final
rules_df = pd.DataFrame(rules_list)
top_10_rules = rules_df.sort_values(by='Lift', ascending=False).head(10)
# FILTRARE PENTRU REGULI UNICE
# Cream o coloana care sorteaza alfabetic perechea pentru a identifica duplicatele "oglinda"
rules_df['pair'] = rules_df.apply(lambda row: str(sorted([row['Antecedent'], row['Consecvent']])), axis=1)

# Pastram doar regula cu Încrederea cea mai mare pentru fiecare pereche unica
unique_rules = rules_df.sort_values('Incredere', ascending=False).drop_duplicates('pair')

# Selectam Top 10 reguli unice după Lift
top_10_unique = unique_rules.sort_values(by='Lift', ascending=False).head(10)

# Salvare
top_10_unique.to_csv('top_10_rules_unice_ro.csv', index=False)

print("Top 10 Reguli UNICE identificate:")
print(top_10_unique[['Antecedent', 'Consecvent', 'Suport', 'Incredere', 'Lift']])

rules_df.to_csv('toate_regulile_asociere.csv', index=False)
