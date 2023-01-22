# Cikk domain klasszifikáció és év prediktálás

Kétféle architektúrát próbáltunk ki az egyik amikor egy kimeneti réteg van és a kimenet típusai a Descartes szorzata a domainnek és az évnek (egy címke lehet a `2004-hvg.hu` például). A másik módszer amikor két kimeneti réteg van, az egyik a domain klasszifikáció a másik pedig egy regresszió az évre.

Az adat alapját a [HunSum-1](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-1) adta, ezen különböző adatszűréseket is végeztünk ehhez a feladathoz specifikusan.

A bemenet minden esetben egy 768 méretű vektor volt ami a [HuBERT](https://huggingface.co/SZTAKI-HLT/hubert-base-cc)-nek volt az egyik kimenete a cikkre. A `HuBERT` minden beadott tokenre egy 768 méretű vektort ad ki, mi ennek 3 féle verzióját használtuk:
- `[CLS]` token klasszifikációs vektorja, ahol a `[CLS]` a szöveg kezdetét jelöli (klasszifikációs vektor az eredeti vektor egy transzformált változata kifejezetten klasszifikációs célra.) (`cls_token`)
- `[CLS]` token vektorja (`start_token`)
- az összes vektor koordinátánkénti átlaga (`avg_token`)

## 1. Egy kimeneti réteggel


### Hiperparaméterek:
- dropout: 0.3
- learning_rate: 0.005
- batch_size: 500000
- hidden_dim: 510
- rejtett rétegek száma: 1

#### csak domain predict a teljes adatra
- 79,6% cls_token
- 81,7% start_token
- 77,4% avg_token

Nagyon nagy adatmennyiségre a domain-year Descartes szorzatnál ~180 osztályunk lett, ilyenkor a model 1% körül teljesített, sok osztályhoz volt hogy 4-5 adatunk volt
-> csökkentettük a lehetséges osztályok számát, azokat hagytuk meg, amelyekben minimum 8000 adatpont található, így 45 prediktálható osztály maradt
Az alábbi eredmények erre a csökkentett adathalmazra vonatkoznak:

#### domain-year
- 26,7% cls_token
- 33,1% avg_token
- 36,3% start_token

#### domain predict
- 84% cls_token

## 1. Két kimeneti réteggel

Az év regressziónál a számok nagysága miatt, standardizáltuk a számokat a `[0, 10]`-es tartományra. Azért volt szükség ilyen nagy tartományra, mert az `MSELoss` nagyon kicsi volt a `CrossEntropyLoss`-hoz képest és az év prediktálás nem javult a tanítás során ha csak a `[0,1]` tartományt használtunk. A regresszió jóságának mérésére az `R2 score`-t használtuk ami az accuracy-hez hasonlóan egy 0-1 közötti értéket ad meg a pontosságra.

### Hiperparaméterek:
- learning_rate: 0.00001
- batch_size: 200
- hidden_dim: 100
- rejtett rétegek száma: 4
- epochs: ?

Az adatból kétfélét próbáltunk ki:
- Az eredeti [HunSum-1](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-1) adatot
- A `HunSum-1` egy olyan változatát ahol kiegyenlítettük a domain-eket és kidobtuk azokat amikből nagyon kevés volt.

### Eredeti `HunSum-1` adat eredményei:

|             | accuracy | R2 score  | cls loss | reg loss | sum loss |
|-------------|----------|-----------|----------|----------|----------|
| cls_token   | x        | x         | x        | x        | x        |
| start_token | x        | x         | x        | x        | x        |
| avg_token   | x        | x         | x        | x        | x        |

### Módosított `HunSum-1` adat eredményei:

|             | accuracy | R2 score  | cls loss | reg loss | sum loss |
|-------------|----------|-----------|----------|----------|----------|
| cls_token   | x        | x         | x        | x        | x        |
| start_token | x        | x         | x        | x        | x        |
| avg_token   | x        | x         | x        | x        | x        |
