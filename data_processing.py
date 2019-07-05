import csv
import pandas as pd

#row proto
with open('1.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[2] for row in reader]

    temp1 = []
    for each in column:
        if each not in temp1:
            temp1.append(each)
    del temp1[0]
    #print(temp)
    #print(len(temp))
    #print(temp[1])

df = pd.read_csv('1.csv')
for i in range(0,len(temp1)):
    df.proto[df['proto']==temp1[i]]=i+1
    #print(i+1)
    #print(temp1[i])
df.to_csv('new.csv', index=False)

#row service
with open('1.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[3] for row in reader]

    temp2 = []
    for each in column:
        if each not in temp2:
            temp2.append(each)
    del temp2[0]

df = pd.read_csv('new.csv')
for i in range(0,len(temp2)):
    df.service[df['service']==temp2[i]]=i+1
    #print(i+1)
    #print(temp2[i])
df.to_csv('new.csv', index=False)

#row state
with open('1.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[4] for row in reader]

    temp3 = []
    for each in column:
        if each not in temp3:
            temp3.append(each)
    del temp3[0]

df = pd.read_csv('new.csv')
for i in range(0,len(temp3)):
    df.state[df['state']==temp3[i]]=i+1
    #print(i+1)
    #print(temp3[i])
df.to_csv('new.csv', index=False)

#row attack_cat
with open('1.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[43] for row in reader]

    temp4 = []
    for each in column:
        if each not in temp4:
            temp4.append(each)
    del temp4[0]

df = pd.read_csv('new.csv')
for i in range(0,len(temp4)):
    df.attack_cat[df['attack_cat']==temp4[i]]=i+1
    #print(i+1)
    #print(temp4[i])
df.to_csv('new.csv', index=False)
