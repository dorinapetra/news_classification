# news_classification

### csak domain predict a teljes adatra
- 79,6% cls_token
- 81,7% start_token

Nagyon nagy adatmennyiségre ~180 osztályunk lett, ilyenkor a model 1% körül teljesített
-> csökkentettük a lehetséges osztályok számát, azokat hagytuk meg, amelyekben minimum 8000 adatpont található, így 45 prediktálható osztály maradt
Az alábbi eredmények erre a csökkentett adathalmazra vonatkoznak:

### domain-year
- 26,7% cls_token
- 33,1% avg_token
- 36,3% start_token

### domain predict
- 84% cls_token
