#!/bin/bash
mkdir -p ../output_new
mkdir -p ../output_new/Born_Again
for n in COMPAS-ProPublica FICO HTRU2 Pima-Diabetes Seeds Breast-Cancer-Wisconsin
do
mkdir -p ../output_new/Born_Again/${n}
done
for o in 4
do
for n in COMPAS-ProPublica FICO HTRU2 Pima-Diabetes Seeds Breast-Cancer-Wisconsin
do
for u in {1..10}
do
for t in 5 10 50 100 250 500
do
./bornAgain ../output_new/RF/$n/${n}.RF$u.T$t.txt ../output_new/Born_Again/${n}/${n}.BA$u.O$o.T$t -trees $t -obj $o
done
done
done
done

