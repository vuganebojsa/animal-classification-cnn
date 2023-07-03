# Animal Classification using Convolutional Neural Networks

## Autor: Nebojsa Vuga SV53/2020

## Tema
# Klasifikacija vrste zivotinje pomocu konvolucione neuronske mreze

## Problem
Problem koji je rešavan je obučavanje konvolucione neuronske mreže za klasifikaciju vrste životinje sa slike.
Implementirana je sekvencijalna neuronska mreža sa 9 skrivenih slojeva, koja treba da raspozna izmedju mačke, psa i divlje životinje.

Rešenje je implementirano u python programskom jeziku uz pomoć sledećih biblioteka: numpy, matplotlib, Pillow, keras, cv2.

## Podaci
Podaci su preuzeti sa sledećeg linka: https://www.kaggle.com/datasets/andrewmvd/animal-faces
Ovaj dataset se sastoji od 16130 slika u 3 klase(psi, mačke, divlje životinje).
Pošto dataset sadrži samo podatke za trening i evaulaciju, programski je podeljen training set na train-test prema odnosu 90%-10%.
Slike su originalno veličine 512x512, ali su programski skalirane na 32x32 radi bolje izgradnje modela i konzistentnosti slika u slučaju drugačijih veličina test slika.
Za trening i test ima ukupno 14630 slika, dok za validaciju ima 1500.

## Pokretanje programa

1. Potrebno je kreirati virtuelno okruzenje komandom py -m venv venv
2. Nakon kreiranje virtuelnog okruzenja potrebno ga je i aktivirati komandom venv/Scripts/activate
3. Nakon uspesne aktivacije potrebno je instalirati biblioteke navedene ispod komandom pip install naziv_biblioteke

Biblioteke:
numpy, 
scikit-learn, 
keras, 
cv2, 
Pillow, 
matplotlib, 

4. Nakon sto su sve biblioteke implementirane moze se pokrenuti projekat komandom py code/main.py

Pre pokretanja imate izbor da otkomentarisete kod ako hocete da istrenirate model, ili da ga ostavite zakomentarisano kako bi on ucitao prethodno sacuvan model sa oko 95% tacnosti.

Kada se pokrene program i ucita model, izlazice vam 10 random slika iz test skupa na ekranu,
koje kada se zatvore daju ispis u konzolu koji predstavlja pretpostavku modela o kojoj je zivotinji rec. 

![](posterSlika.png?raw=true)
