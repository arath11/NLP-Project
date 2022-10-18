from nltk import *
from nltk.corpus import stopwords

stop_words=set(stopwords.words('english'))
print(stop_words)

# con4= FreqDist([i for i in string])#if len(i) == 4
# reg=sorted(i for i in string if con4[i]<5000 and con4[i]>5)
# limpio=[i for i in string if not i.lower() in stop_words]
# print(f'len: {len(limpio)}')