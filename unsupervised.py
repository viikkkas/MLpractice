

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


dataset=[['Milk', 'Butter', 'Bread', 'Eggs'],
         ['Onion', 'Potato',  'Eggs', 'Salt'],
         ['Milk', 'Bread', 'Potato', 'Salt'],
         ['Milk', 'Bread', 'Potato', 'Salt', 'Onion'],
         ['Butter', 'Bread', 'Eggs', 'Milk']]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


print(apriori(df, min_support=0.6, use_colnames=True))
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
print(frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.6) ])
print(frequent_itemsets[ frequent_itemsets['itemsets'] == {'Onion', 'Eggs'} ])

