#chcp 65001
from bs4 import BeautifulSoup
import os, sys
import urllib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier


train = [
        ('1.000 Folgen: Das Quiz zum Tatort-Jubiläum', 'header'),
        ('1.000 Tatorte in Berlin mit Schimanskis blutbefleckter Jacke', 'header'),
        ('1. April: Photobombs with David Hasselhoff und Hofer-Roboter', 'header'),
        ('Zehn Jahre Comic-Vertrieb Pictopia: Eine One-Man-Show', 'header'),
        ('Zehn-Minuten-Küche: Dem Fisch Dampf machen', 'header'),
        ('Zehn spektakuläre Pools mit Aussicht', 'header'),
        ('11.500 Wahlberechtigte mehr als im April und Mai', 'header'),
        ('13-Jähriger angeschossen: Prozess um versuchten Mord in Wien', 'header'),
        ('13 Tote durch Brand in Lederfabrik nahe Neu Delhi', 'header'),
        ('14 Kilometer langer Tunnel durch die Anden geplant', 'header'),
        ('14 Tipps für Hausstaubmilbenallergiker', 'header'),
        ('Gewaltvideo: Polizei nimmt Hauptverdächtige fest, neuer Fall in Wien', 'header'),
        ('3,2 Milliarden Jahre alte Lebensspuren in Südafrika entdeckt', 'header'),
        ('Schlaflabor: 3D-videoüberwacht im Bett', 'header'),
        ('5th Avenue teuerste Shoppingmeile der Welt, Kohlmarkt auf Platz zehn', 'header'),
        ('5th Avenue teuerste Shoppingmeile der Welt, Kohlmarkt Platz zehn', 'header'),
        ('Sechs Tipps für die gesunde Sauna', 'header'),
        ('Sieben Risikofaktoren für Alzheimer', 'header'),
        ('Acht Tipps für "Fit durch den Winter"', 'header'),
        ('Neu-Delhi – In der Nähe der indischen Hauptstadt Neu Delhi sind am Freitag durch einen Brand in einer Lederfabrik 13 Menschen ums Leben gekommen. Bei den Toten handelte es sich nach Polizeiangaben um Beschäftigte des Unternehmens, das sich in der östlichen Vorstadt Sahibabad befindet. Die Arbeiter hatten demnach in der Firma übernachtet und wurden von dem Brand im Schlaf überrascht. Zwei oder drei Arbeiter seien verletzt ins Krankenhaus gebracht worden, sagte Polizeisprecher Bhagwat Singh.', 'NOT'),
        ('In Indien gibt es immer wieder schwere Unfälle an Arbeitsstätten. Im Oktober kamen bei einer Explosion in einem Unternehmen für Feuerwerkskörper im Bundesstaat Tamil Nadu acht Arbeiter ums Leben. Bei einem ähnlichen Unglück gab es im Mai 2014 im Bundesstaat Madhya Pradesh 15 Tote. (APA, 11.11.2016)', 'NOT')

]

cl = NaiveBayesClassifier(train)
file=open("test.txt");
t=file.read();
blob = TextBlob(t, classifier=cl)
print(blob)
print(blob.classify())
for sentence in blob.sentences:
    print(sentence)
    print(sentence.classify())

# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))

# Show 5 most informative features
cl.show_informative_features()
