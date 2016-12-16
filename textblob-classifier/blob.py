from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier


#Custom training data with NaiveBayes
train = [
    ('Buy cheap drugs', 'spam'),
    ('Cheap viagra', 'spam'),
    ('Win 1000 dollar', 'spam'),
    ('Greatings, how are you?', 'ham'),
    ('What an awesome picture', 'ham'),
    ('Send me your adress', 'ham'),
    ('viagra', 'spam')
]
test = [
    ('Cheap drugs', 'spam'),
    ('Buy viagra', 'spam'),
    ("Send me your picture", 'ham'),
    ("Provide you CV", 'ham')
]

cl = NaiveBayesClassifier(train)

# Classify some text
print(cl.classify("Buy drugs"))  # "spam"
print(cl.classify("How are you?"))   # "ham"

# Classify a TextBlob
blob = TextBlob("Buy cheap drugs for only 100 dollar. Send me your adress and you will get viagra for free!", classifier=cl)
print(blob)
print(blob.classify())

for sentence in blob.sentences:
    print(sentence)
    print(sentence.classify())

# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))

# Show 5 most informative features
cl.show_informative_features()


#Classifying txt file with DecisionTree

train = [
    ('Buy cheap drugs', 'spam'),
    ('Cheap viagra', 'spam'),
    ('Win 1000 dollar', 'spam'),
    ('Greatings, how are you?', 'ham'),
    ('What an awesome picture', 'ham'),
    ('Send me your adress', 'ham'),
    ('viagra', 'spam')
]

tr = DecisionTreeClassifier(train)

file=open("test.txt");
t=file.read();
print(type(t))
blob = TextBlob(t, classifier=tr)
blob.tags
print(blob)

for sentence in blob.sentences:
    print(sentence)
    print(sentence.classify())

print(blob.classify())
print(tr.pseudocode())
print(tr.pretty_format())
print(tr.pprint())
