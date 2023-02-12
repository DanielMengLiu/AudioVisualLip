file_a_mis = 'a_misclassification.txt'
file_v_mis = 'v_misclassification.txt'
file_av_mis = 'av_misclassification.txt'

lst_a = []
lst_v = []
lst_av = []

with open(file_a_mis, 'r') as a:
    for line in a:
        line = line.rstrip()
        utt, label, score = line.split(' ')
        lst_a.append(utt)
with open(file_v_mis, 'r') as v:
    for line in v:
        line = line.rstrip()
        utt, label, score = line.split(' ')
        lst_v.append(utt)
with open(file_av_mis, 'r') as av:
    for line in av:
        line = line.rstrip()
        utt, label, score = line.split(' ')
        lst_av.append(utt)
        
bothwrong = list(set(lst_a).intersection(set(lst_v)))
print(len(bothwrong))

aaa = list(set(lst_a).union(set(lst_v)))
print(len(aaa))
