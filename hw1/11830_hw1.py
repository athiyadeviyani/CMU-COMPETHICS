#!/usr/bin/env python
# coding: utf-8

# # 11-830 Computational Ethics for NLP
# ## Homework 1

# In[422]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import os
from tqdm import tqdm

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# # Preprocessing

# ## Load the data

# In[423]:


train = pd.read_table('snli/snli_1.0_train.txt', delimiter = '\t')


# In[424]:


train.head()


# In[425]:


premises = pd.DataFrame()
premises["sentence"] = train["sentence1"]

hypotheses = pd.DataFrame()
hypotheses["sentence"] = train["sentence2"]


# In[426]:


premises.head()


# In[427]:


hypotheses.head()


# In[428]:


my_file = open('identity_labels.txt', "r")
content = my_file.read()
identities = content.split('\n')
identities[:10]


# ## Preprocess the data
# 
# Lowercase, remove stopwords and tokenize the data. Remove duplicates.

# In[429]:


premises.drop_duplicates(inplace=True)
hypotheses.drop_duplicates(inplace=True)


# In[430]:


premises.describe()


# In[431]:


hypotheses.describe()


# In[432]:


tokenizer = RegexpTokenizer('[a-z]\w+')

stop_words = [x for x in stop_words if x not in identities]

def preprocess(sentence):
    if isinstance(sentence, float):
        return []
    else:
        sentence = sentence.lower()
        tokenized = tokenizer.tokenize(sentence)
        return list(set([word for word in tokenized if word not in stop_words]))
    
preprocess('Person on a horse jumps over a broken down airplane.')


# In[433]:


# This cell takes a while to run (~2 mins)

premises["sentence"] = premises["sentence"].map(preprocess)
hypotheses["sentence"] = hypotheses["sentence"].map(preprocess)


# In[434]:


hypotheses.head(1), premises.head(1)


# ## Get the word occurence dictionary for premise and hypothesis

# In[435]:


premise_dict = {}
hyp_dict = {}

def fill_dict(source, target_dict):
    for sentence in tqdm(source.values):
        sentence = sentence[0]
        for word in sentence:
            target_dict[word] = target_dict.get(word, 0) + 1
            
fill_dict(premises, premise_dict)
fill_dict(hypotheses, hyp_dict)


# In[436]:


len(premise_dict), len(hyp_dict)


# In[437]:


hyp_dict['skimpy']


# In[438]:


premise_dict = {k:v for k,v in premise_dict.items() if v >= 10}
hyp_dict = {k:v for k,v in hyp_dict.items() if v >= 10}


# In[439]:


len(premise_dict), len(hyp_dict)


# In[440]:


co_premise_dict = {}
co_hyp_dict = {}

def fill_co_dict(source, source_dict, target_dict):
    for sentence in tqdm(source.values):
        sentence = sentence[0]
        for identity in identities:
            if identity in sentence:
                for word in sentence:
                    if word in source_dict:
                        key = (identity, word)
                        target_dict[key] = target_dict.get(key, 0) + 1
                    
fill_co_dict(premises, premise_dict, co_premise_dict)
fill_co_dict(hypotheses, hyp_dict, co_hyp_dict)


# In[441]:


len(co_premise_dict), len(co_hyp_dict)


# In[442]:


key = ('woman', 'man')
co_premise_dict[key]


# In[443]:


sorted_coprem = sorted(co_premise_dict.items(), key=lambda x: x[1], reverse=True)
[((x,y), z) for ((x,y),z) in sorted_coprem if x != y][:10]


# In[448]:


sorted_cohyp = sorted(co_hyp_dict.items(), key=lambda x: x[1], reverse=True)
[((x,y), z) for ((x,y),z) in sorted_cohyp if x != y and x in identities and y not in identities][:10]


# In[445]:


# co_premise_dict = {k:v for k,v in co_premise_dict.items() if v >= 10}
# co_hyp_dict = {k:v for k,v in co_hyp_dict.items() if v >= 10}


# In[449]:


[(k,v) for (k,v) in co_hyp_dict.items() if k[0] == "she"]


# In[450]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'shooting' == j], key=lambda x: x[1], reverse=True)


# In[451]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'thieves' == j], key=lambda x: x[1], reverse=True)


# In[452]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'prostitutes' == j], key=lambda x: x[1], reverse=True)


# ## Calculte PMI

# In[453]:


def get_pmi(wi, wj, source="premise"):
    if source == "premise":
        word_count_dic = premise_dict
        co_dict = co_premise_dict
        N = len(premises.values)
    else:
        word_count_dic = hyp_dict
        co_dict = co_hyp_dict
        N = len(hypotheses.values)
    
    c_wi = word_count_dic[wi]
    c_wj = word_count_dic[wj]
    c_wi_wj = co_dict[(wi, wj)]
    # print(c_wi, c_wj, c_wi_wj)
    
    pmi = np.log2( (N * c_wi_wj) / (c_wi * c_wj) )
    
    return pmi


# In[454]:


get_pmi('woman', 'man', source="premise")


# In[455]:


def get_pmi_all(source="premise"):
    if source == "premise":
        co_dict = co_premise_dict
        word_count_dic = premise_dict
    else:
        co_dict = co_hyp_dict
        word_count_dic = hyp_dict
    
    pmi_dict = {}
    for (identity, word) in co_dict:
        if identity in word_count_dic and word in word_count_dic:
            key = (identity, word)
            pmi_dict[key] = get_pmi(identity, word, source)
        
    return pmi_dict


# In[456]:


pmi_premise = get_pmi_all("premise")


# In[457]:


pmi_hyp = get_pmi_all("hyp")


# In[458]:


sorted_pmiprem = sorted(pmi_premise.items(), key=lambda x: x[1], reverse=True)
[((x,y), z) for ((x,y),z) in sorted_pmiprem if x != y][:20]


# In[459]:


sorted_pmihyp = sorted(pmi_hyp.items(), key=lambda x: x[1], reverse=True)
[((x,y), z) for ((x,y),z) in sorted_pmihyp if x != y][:20]


# # Analysis

# In[460]:


def get_n_highest(id_word, pmi_dict, n=10, plot=False, rotation=45, figsize=(20, 3)):
    id_list = [((x,y),z) for ((x,y),z) in pmi_dict.items() if x==id_word]
    sorted_list = sorted(id_list, key=lambda x: x[1], reverse=True)[:n]
    
    if pmi_dict == pmi_premise or pmi_dict == pmi_conprem:
        title = "premises"
    else:
        title = "hypotheses"
    
    if plot:
        plt.figure(figsize=figsize)
        plt.bar([x[0][1] for x in sorted_list], [x[1] for x in sorted_list])
        plt.xticks(rotation=rotation)
        plt.xlabel("corpus word")
        plt.ylabel("PMI")
        plt.title("Highest PMIs for the identity:" + id_word + " in " + title)
        plt.show()
        
    else:
        return sorted_list


# In[412]:


get_n_highest('woman', pmi_premise, n=25, plot=True)


# In[413]:


get_n_highest('woman', pmi_hyp, n=25, plot=True)


# In[462]:


get_n_highest('she', pmi_premise, n=25, plot=True)


# In[463]:


get_n_highest('she', pmi_hyp, n=25, plot=True)


# In[465]:


find_sentence("she", "strip", "hyp")


# In[466]:


pd.set_option('display.max_rows', 1000)


# In[467]:


p_availabilities = {}

for identity in identities:
    p_availabilities[identity] = len(get_n_highest(identity, pmi_premise, n=10000, plot=False))


# In[468]:


h_availabilities = {}

for identity in identities:
    h_availabilities[identity] = len(get_n_highest(identity, pmi_hyp, n=10000, plot=False))


# In[41]:


def compare(w1, w2, source):
    df = pd.DataFrame()
    
    intersection =         [b for ((a,b), c) in get_n_highest(w1, source, n=10000, plot=False)          if b in [y for ((x,y),z) in get_n_highest(w2, source, n=10000, plot=False)]]

    words = []
    pmis_w1 = []
    pmis_w2 = []

    for word in intersection:
        pmi_w1 = source[(w1, word)]
        pmi_w2 = source[(w2, word)]

        words.append(word)
        pmis_w1.append(pmi_w1)
        pmis_w2.append(pmi_w2)

    df["words"] = words
    df["pmi_" + w1] = pmis_w1
    df["pmi_" + w2] = pmis_w2
    df["diff"] = df["pmi_" + w1] - df["pmi_" + w2]
    
    return df


# In[397]:


def find_sentence(w1, w2="", source="premise"):
    if source=="premise":
        sent = "sentence1"
    else:
        sent = "sentence2"
        
    res = set()
    count = 0
    for sentence in train[sent].values:
        if isinstance(sentence, float):
            continue
        if w1 in sentence and w2 in sentence:
            count += 1                
            res.add(sentence)
            
    return res


# In[43]:


list(find_sentence("asian", "girl", "hypothesis"))[:10]


# ### Comparison between female-male

# In[44]:


female_male_prem = compare('female', 'male', source=pmi_premise)


# In[45]:


female_male_prem.sort_values("diff", ascending=False).head(50)


# In[46]:


female_male_prem.sort_values("diff", ascending=True).head(50)


# In[47]:


female_male_hyp = compare('female', 'male', source=pmi_hyp)


# In[48]:


female_male_hyp.sort_values("diff", ascending=False).head(50)


# In[49]:


female_male_hyp.sort_values("diff", ascending=True).head(50)


# ### Comparison between woman-man

# In[50]:


woman_man = compare('woman', 'man', source=pmi_premise)


# In[51]:


woman_man.sort_values("pmi_woman", ascending=False).head(5)


# In[52]:


woman_man.sort_values("pmi_man", ascending=False).head(5)


# In[53]:


woman_man[woman_man["words"] == "topless"]


# In[54]:


woman_man[woman_man["words"] == "dances"]


# In[55]:


woman_man[woman_man["words"] == "football"]


# In[56]:


woman_man.sort_values("diff", ascending=False).head(50)


# In[57]:


# soccer, baseball, basketball, football


# In[58]:


woman_man.sort_values("diff", ascending=True).head(50)


# In[59]:


woman_man_hyp = compare('woman', 'man', source=pmi_hyp)


# In[60]:


woman_man_hyp.sort_values("diff", ascending=True).head(50)


# In[61]:


woman_man_hyp.sort_values("diff", ascending=False).head(50)


# ### Comparison between women-men

# In[62]:


women_men_prem = compare('women', 'men', source=pmi_premise)


# In[63]:


women_men_prem.sort_values("diff", ascending=False).head(50)


# In[64]:


women_men_prem.sort_values("diff", ascending=True).head(50)


# In[65]:


women_men_hyp = compare('women', 'men', source=pmi_hyp)


# In[66]:


women_men_hyp.sort_values("diff", ascending=True).head(50)


# In[67]:


women_men_hyp.sort_values("diff", ascending=False).head(50)


# ### Comparison between girl-boy

# In[68]:


girl_boy_prem = compare('girl', 'boy', source=pmi_premise)


# In[69]:


girl_boy_prem.sort_values("diff", ascending=False).head(50)


# In[70]:


girl_boy_prem.sort_values("diff", ascending=True).head(50)


# In[71]:


girl_boy_hyp = compare('girl', 'boy', source=pmi_hyp)


# In[72]:


girl_boy_hyp.sort_values("diff", ascending=True).head(50)


# In[73]:


girl_boy_hyp.sort_values("diff", ascending=False).head(50)


# ### Comparison between girls-boys

# In[74]:


girls_boys_prem = compare('girls', 'boys', source=pmi_premise)


# In[75]:


girls_boys_prem.sort_values("diff", ascending=False).head(50)


# In[76]:


girls_boys_prem.sort_values("diff", ascending=True).head(50)


# In[77]:


girls_boys_hyp = compare('girls', 'boys', source=pmi_hyp)


# In[78]:


girls_boys_hyp.sort_values("diff", ascending=False).head(50)


# In[79]:


girls_boys_hyp.sort_values("diff", ascending=True).head(50)


# ### Comparison between white/caucasian-black/african

# In[80]:


white_black_prem = compare('white', 'black', source=pmi_premise)
white_black_hyp = compare('white', 'black', source=pmi_hyp)

cauc_af_prem = compare('caucasian', 'african', source=pmi_premise)
cauc_af_hyp = compare('caucasian', 'african', source=pmi_hyp)


# In[81]:


white_black_prem.sort_values("diff", ascending=False).head(50)


# In[82]:


white_black_prem.sort_values("diff", ascending=True).head(50)


# In[83]:


white_black_hyp.sort_values("diff", ascending=False).head(50)


# In[84]:


white_black_hyp.sort_values("diff", ascending=True).head(50)


# #### Caucasian/African

# In[85]:


cauc_af_prem.sort_values("diff", ascending=False).head(50)


# In[86]:


cauc_af_hyp.sort_values("diff", ascending=False).head(50)


# ### Comparison between white/caucasian-asian

# In[87]:


white_asian_prem = compare('white', 'asian', source=pmi_premise)
white_asian_hyp = compare('white', 'asian', source=pmi_hyp)

cauc_asian_prem = compare('caucasian', 'asian', source=pmi_premise)
cauc_asian_hyp = compare('caucasian', 'asian', source=pmi_hyp)


# In[88]:


white_asian_prem.sort_values("diff", ascending=False).head(50)


# In[89]:


white_asian_prem.sort_values("diff", ascending=True).head(50)


# In[90]:


white_asian_hyp.sort_values("diff", ascending=False).head(50)


# In[91]:


white_asian_hyp.sort_values("diff", ascending=True).head(50)


# In[92]:


cauc_asian_prem.sort_values("diff", ascending=False).head(50)


# In[93]:


cauc_asian_hyp.sort_values("diff", ascending=True).head(50)


# ### Comparison between white/caucasian-indian

# In[94]:


white_indian_prem = compare('white', 'indian', source=pmi_premise)
white_indian_hyp = compare('white', 'indian', source=pmi_hyp)

cauc_indian_prem = compare('caucasian', 'indian', source=pmi_premise)
cauc_indian_hyp = compare('caucasian', 'indian', source=pmi_hyp)


# In[95]:


white_indian_prem.sort_values("diff", ascending=True).head(50)


# In[96]:


white_indian_prem.sort_values("diff", ascending=False).head(50)


# In[97]:


white_indian_hyp.sort_values("diff", ascending=True).head(50)


# In[98]:


white_indian_hyp.sort_values("diff", ascending=False).head(50)


# In[99]:


cauc_indian_prem.sort_values("diff", ascending=True).head(50)


# In[100]:


cauc_indian_hyp.sort_values("diff", ascending=True).head(50)


# ### Comparison between asian-indian

# In[101]:


asian_indian_prem = compare('asian', 'indian', source=pmi_premise)
asian_indian_hyp = compare('asian', 'indian', source=pmi_hyp)


# In[102]:


asian_indian_prem.sort_values("diff", ascending=True).head(50)


# In[103]:


get_n_highest('chinese', pmi_hyp, n=60, plot=True)


# ### Comparison between young-old/elderly

# In[104]:


young_old_prem = compare('young', 'old', source=pmi_premise)
young_old_hyp = compare('young', 'old', source=pmi_hyp)

young_eld_prem = compare('young', 'elderly', source=pmi_premise)
young_eld_hyp = compare('young', 'elderly', source=pmi_hyp)


# In[105]:


young_old_prem.sort_values("diff", ascending=True).head(50)


# In[106]:


young_old_prem.sort_values("diff", ascending=False).head(50)


# In[107]:


young_old_hyp.sort_values("diff", ascending=True).head(50)


# In[108]:


young_old_hyp.sort_values("diff", ascending=False).head(50)


# In[109]:


young_eld_prem.sort_values("diff", ascending=False).head(50)


# In[110]:


young_eld_prem.sort_values("diff", ascending=True).head(50)


# In[111]:


young_eld_hyp.sort_values("diff", ascending=False).head(50)


# In[112]:


young_eld_hyp.sort_values("diff", ascending=True).head(50)


# In[ ]:





# In[ ]:





# # POPULAR STEREOTYPES ANALYSIS

# ### Sports

# In[113]:


get_pmi("woman", "player"), get_pmi("man", "player")


# In[114]:


get_pmi("woman", "football"), get_pmi("man", "football")


# In[115]:


get_pmi("woman", "basketball"), get_pmi("man", "basketball")


# In[116]:


get_pmi("woman", "sports"), get_pmi("man", "sports")


# In[117]:


get_pmi("girl", "sports"), get_pmi("boy", "sports")


# In[118]:


get_pmi("girls", "baseball"), get_pmi("boys", "baseball")


# In[119]:


get_pmi("girl", "baseball"), get_pmi("boy", "baseball")


# ### Are women really #domesticated ?

# In[120]:


get_pmi("woman", "kitchen"), get_pmi("man", "kitchen")


# In[121]:


get_pmi("woman", "groceries"), get_pmi("man", "groceries")


# In[122]:


get_pmi("woman", "laundry"), get_pmi("man", "laundry")


# In[123]:


get_pmi("woman", "ironing")


# In[124]:


get_pmi("woman", "dishwasher")


# In[125]:


get_pmi("woman", "apron"), get_pmi("man", "apron")


# In[126]:


get_pmi("woman", "sewing"), get_pmi("man", "sewing")


# In[127]:


get_pmi("woman", "knitting")


# ### School

# In[128]:


get_pmi("woman", "children"), get_pmi("man", "children")


# In[129]:


get_pmi("woman", "school"), get_pmi("man", "school")


# In[130]:


get_pmi("girls", "school"), get_pmi("boys", "school")


# In[131]:


get_pmi("girl", "school"), get_pmi("boy", "school")


# In[132]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'school' == j], key=lambda x: x[1], reverse=True)


# In[133]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'graduation' == j], key=lambda x: x[1], reverse=True)


# In[134]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'studies' == j], key=lambda x: x[1], reverse=True)


# ### Clothing

# In[135]:


get_pmi("woman", "blouse"), get_pmi("man", "shirt")


# In[ ]:





# In[ ]:





# ### Occupation

# In[136]:


pmi_premise[("man", "construction")], pmi_premise[("woman", "construction")]


# In[137]:


pmi_premise[("man", "hospital")], pmi_premise[("woman", "construction")]


# In[138]:


pmi_premise[("man", "working")], pmi_premise[("woman", "working")]


# In[139]:


pmi_premise[("man", "work")], pmi_premise[("woman", "work")]


# In[140]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'work' == j], key=lambda x: x[1], reverse=True)


# In[ ]:





# In[ ]:





# ### Young people/women are attractive

# In[141]:


get_pmi("young", "attractive")


# In[142]:


get_pmi("woman", "attractive")


# In[143]:


get_pmi("woman", "beautiful")


# In[ ]:





# In[ ]:





# ### Asians are smart, dancy, traditional and mostly women?

# In[144]:


get_pmi("asian", "school")


# In[145]:


get_pmi("asian", "restaurant")


# In[146]:


get_pmi("indian", "dancing")


# In[147]:


get_pmi("asian", "traditional"), get_pmi("indian", "traditional")


# In[148]:


get_pmi("asian", "woman"), get_pmi("asian", "man")


# In[149]:


get_pmi("indian", "woman"), get_pmi("indian", "man")


# ### women are more likely to be described by their physique especially if they are not white

# In[150]:


get_pmi("asian", "woman"), get_pmi("asian", "man")


# In[151]:


get_pmi("indian", "woman"), get_pmi("indian", "man")


# In[152]:


get_pmi("caucasian", "woman"), get_pmi("caucasian", "man")


# In[153]:


get_pmi("woman", "attractive")


# In[194]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'attractive' == j], key=lambda x: x[1], reverse=True)


# In[195]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'attractive' == j], key=lambda x: x[1], reverse=True)


# In[154]:


# get_pmi("man", "handsome") does not exist


# In[155]:


get_pmi("woman", "beautiful")


# In[156]:


get_pmi("woman", "blouse"), get_pmi("man", "shirt")


# In[157]:


get_n_highest("woman", pmi_premise, n=50, plot=True, rotation=80, figsize=(15, 3))


# In[158]:


get_pmi("woman", "short")


# In[159]:


get_pmi("man", "short")


# In[160]:


get_n_highest("man", pmi_premise, n=200, plot=False, rotation=80, figsize=(15, 3))


# ### Man doing manly things

# In[185]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'weapon' == j], key=lambda x: x[1], reverse=True)


# In[186]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'knife' == j], key=lambda x: x[1], reverse=True)


# In[187]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'aiming' == j], key=lambda x: x[1], reverse=True)


# In[188]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'operates' == j], key=lambda x: x[1], reverse=True)


# In[189]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'hammer' == j], key=lambda x: x[1], reverse=True)


# In[190]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'inspects' == j], key=lambda x: x[1], reverse=True)


# In[191]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'welds' == j], key=lambda x: x[1], reverse=True)


# In[192]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'sculpting' == j], key=lambda x: x[1], reverse=True)


# In[193]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'rifle' == j], key=lambda x: x[1], reverse=True)


# In[170]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'blowtorch' == j], key=lambda x: x[1], reverse=True)


# ### Men smile less

# In[171]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'smiling' == j], key=lambda x: x[1], reverse=True)


# In[ ]:





# # Hypotheses Analysis

# In[172]:


get_n_highest("women", pmi_hyp, n=50, plot=True, rotation=60, figsize=(15, 3))


# In[173]:


get_n_highest("women", pmi_premise, n=50, plot=True, rotation=60, figsize=(15, 3))


# In[175]:


get_n_highest("asian", pmi_premise, n=50, plot=True, rotation=60, figsize=(15, 3))


# In[174]:


get_n_highest("asian", pmi_hyp, n=50, plot=True, rotation=60, figsize=(15, 3))


# In[180]:


get_n_highest("asian", pmi_hyp, n=100, plot=False, rotation=60, figsize=(15, 3))


# In[207]:


get_n_highest("asian", pmi_premise, n=100, plot=False, rotation=60, figsize=(15, 3))


# In[208]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'gay' == i], key=lambda x: x[1], reverse=True)


# In[212]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'christian' == i], key=lambda x: x[1], reverse=True)


# In[211]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'hell' == j], key=lambda x: x[1], reverse=True)


# In[214]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'muslim' == i], key=lambda x: x[1], reverse=True)


# In[215]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'muslim' == i], key=lambda x: x[1], reverse=True)


# In[217]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'jewish' == i], key=lambda x: x[1], reverse=True)


# In[219]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'jewish' == i], key=lambda x: x[1], reverse=True)


# In[177]:


def find_sentence(w1, w2, source="premise"):
    if source=="premise":
        sent = "sentence1"
    else:
        sent = "sentence2"
        
    res = set()
    count = 0
    for sentence in train[sent].values:
        if isinstance(sentence, float):
            continue
        if w1 in sentence and w2 in sentence:
            count += 1                
            res.add(sentence)
            
    return res


# In[178]:


[x for x in train["sentence2"] if not isinstance(x, float) and "asian" in x and "skimpy" in x]


# # PMI ANALYSIS - PREMISE
# 
# **Gender**
# - male/female
# - man/woman
# - men/women
# - girl/boy
# 
# **Race**
# - caucasian
# - asian
# - indian
# - chinese
# - black
# - white
# - african
# 
# **Religion**
# - christian/s
# - muslim/s
# - jewish/jews
# - buddhist/s
# - sikh/s
# 
# **Age**
# - young
# - old
# - elderly

# ## Gender
# 
# - male/female
# - man/woman
# - men/women
# - girl/boy

# ### hypothesis - male/female

# In[260]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'male' == i], key=lambda x: x[1], reverse=True)[:100]


# In[261]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'female' == i], key=lambda x: x[1], reverse=True)[:100]


# Observations
# - male: sripper, perusing, fooling, chief, determined, protestor, conversating, ghetto, con
# - female: slender, tummy, bra, revealing, servers, asssists, sew, youthful, protestor

# ### hypothesis - man/woman

# In[262]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'man' == i], key=lambda x: x[1], reverse=True)[:100]


# In[263]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'woman' == i], key=lambda x: x[1], reverse=True)[:100]


# Observation
# - man: robs, boom, dragged, flirts, interviews, machete, operates, homeless, commiting, chainsaw, creepy, insane
# - woman: bra, shawl, headscarf, injection, pregnant, panties, attractive, skirt, flirts, bikini, sexy, skimpy, lingerie, babysitting, knitting, flirting

# ### hypothesis - men/women

# In[264]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'men' == i], key=lambda x: x[1], reverse=True)[:100]


# In[265]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'women' == i], key=lambda x: x[1], reverse=True)[:100]


# Observations
# - men: gutters, inappropriate, laptops, chemical, seminar, politics, dueling, swords, weapons, prefessionally, wrangle, mission, argue, discuss
# - women: gossip, bikinis, gossiping, flirt, scantily, beauty, necklaces

# ### hypothesis - girl/boy

# In[266]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'girl' == i], key=lambda x: x[1], reverse=True)[:100]


# In[267]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'boy' == i], key=lambda x: x[1], reverse=True)[:100]


# nothing interesting

# ## Race
# 
# - caucasian
# - asian
# - indian
# - chinese
# - black
# - white
# - african

# ### hypothesis - caucasian

# In[270]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'caucasian' == i], key=lambda x: x[1], reverse=True)[:100]


# caucasian: handsome, taught, purchases, instructor, nightclub, americaan, impress, teacher, rugby, blond, gentleman, protesting

# ### hypothesis - asian

# In[275]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'asian' == i], key=lambda x: x[1], reverse=True)[:1000]


# asian: skimpy, disaster, elders, slender, knowledge, tradition, prostitutes, intoxicated, literature, playboy, employment, ceo, rowdy, 

# In[276]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'asians' == i], key=lambda x: x[1], reverse=True)[:1000]


# In[274]:


[((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'asian' == i and j == 'ceo']


# asians: tradition, devoid, explain, mourning, airplanes, killing, everwhere, herd, exploring, china, chinese, company

# ### hypothesis - indian

# In[281]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'indian' == i], key=lambda x: x[1], reverse=True)[:1000]


# indian: passionately, heritage, saris, warrior, kidnapping, ritual, dmv, murder, serious, attacks, scary

# In[278]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'murder' == j], key=lambda x: x[1], reverse=True)[:100]


# In[280]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'attacks' == j], key=lambda x: x[1], reverse=True)[:100]


# In[282]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'scary' == j], key=lambda x: x[1], reverse=True)[:100]


# ### hypothesis - chinese

# In[284]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'chinese' == i], key=lambda x: x[1], reverse=True)[:1000]


# chinese: aloud, competed, employyment, garb, smashes, celebration, target, burned, asians, lazy, shooting

# In[285]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'lazy' == j], key=lambda x: x[1], reverse=True)[:100]


# In[287]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'shooting' == j], key=lambda x: x[1], reverse=True)[:100]


# ### hypothesis - black

# In[288]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'black' == i], key=lambda x: x[1], reverse=True)[:1000]


# black: ghetto, digger, stripped, footballer, sexy, robbed, thieves, satan, viciously, commit

# In[301]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'robbers' in j], key=lambda x: x[1], reverse=True)[:100]


# In[302]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'robbed' in j], key=lambda x: x[1], reverse=True)[:100]


# In[291]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'thieves' == j], key=lambda x: x[1], reverse=True)[:100]


# In[292]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'satan' == j], key=lambda x: x[1], reverse=True)[:100]


# In[298]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'viciously' in j], key=lambda x: x[1], reverse=True)[:100]


# In[299]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'commit' in j], key=lambda x: x[1], reverse=True)[:100]


# In[297]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'arrest' in j], key=lambda x: x[1], reverse=True)[:100]


# ### hypothesis - white

# In[304]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'white' == i], key=lambda x: x[1], reverse=True)[:100]


# ### hypothesis - african

# In[305]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'african' == i], key=lambda x: x[1], reverse=True)[:100]


# In[306]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'africans' == i], key=lambda x: x[1], reverse=True)[:100]


# african/s: guns, poverty, ghetto, jaywalking, secluded, warrior, die, forced

# In[307]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'gun' in j], key=lambda x: x[1], reverse=True)[:100]


# In[310]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'dies' == j], key=lambda x: x[1], reverse=True)[:100]


# ## Religion
# - christian/s
# - muslim/s
# - jewish/jews
# - buddhist/s
# - sikh/s

# ### hypothesis - christian/s

# In[313]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'christian' in i], key=lambda x: x[1], reverse=True)[:100]


# christian: strips, church, home, holidays, christmas

# ### hypothesis - muslim/s

# In[312]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'muslim' in i], key=lambda x: x[1], reverse=True)[:100]


# In[314]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'weapon' == j], key=lambda x: x[1], reverse=True)[:100]


# In[315]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'dead' == j], key=lambda x: x[1], reverse=True)[:100]


# In[317]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'terrorist' in j], key=lambda x: x[1], reverse=True)[:100]


# muslim/s: erroorists, weapon, temple, headed, laying, dead, body

# ### hypothesis - jewish/jews

# In[318]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'jewish' == i], key=lambda x: x[1], reverse=True)[:100]


# In[328]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'armed' in j], key=lambda x: x[1], reverse=True)[:100]


# ### hypothesis - buddhist/s

# In[319]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'buddhist' == i], key=lambda x: x[1], reverse=True)[:100]


# ### hypothesis - sikh/s

# In[255]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'sikh' == i], key=lambda x: x[1], reverse=True)[:100]


# ## Age
# - young
# - old
# - elderly

# ### hypothesis - young

# In[320]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'young' == i], key=lambda x: x[1], reverse=True)[:100]


# In[321]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'old' == i], key=lambda x: x[1], reverse=True)[:100]


# In[322]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'elderly' == i], key=lambda x: x[1], reverse=True)[:100]


# In[323]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'teenage' in i], key=lambda x: x[1], reverse=True)[:100]


# In[324]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'attractive' == j], key=lambda x: x[1], reverse=True)[:100]


# In[325]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'sexy' == j], key=lambda x: x[1], reverse=True)[:100]


# In[327]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'protest' in j], key=lambda x: x[1], reverse=True)[:100]


# # PMI ANALYSIS - PREMISE
# 
# **Gender**
# - male/female
# - man/woman
# - men/women
# - girl/boy
# 
# **Race**
# - caucasian
# - asian
# - indian
# - chinese
# - black
# - white
# - african
# 
# **Religion**
# - christian/s
# - muslim/s
# - jewish/jews
# - buddhist/s
# - sikh/s
# 
# **Age**
# - young
# - old
# - elderly

# ## Gender

# ### premise - male/female

# In[226]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'male' == i], key=lambda x: x[1], reverse=True)[:100]


# In[227]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'female' == i], key=lambda x: x[1], reverse=True)[:100]


# Observation: male has things like surgeon, dentist, chief, scientists while women have more artsy stuff like violinist guitarist, designer. male also has these but lower.

# ### premise - man/woman

# In[228]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'man' == i], key=lambda x: x[1], reverse=True)[:100]


# In[229]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'woman' == i], key=lambda x: x[1], reverse=True)[:100]


# Observation: man has tools like sledgehammer, woodworking, chainsaw, rifle, while women has a lot of clothes-related stuff like bra, handbag ,blouse and even things like revealing. ironing and dishwasher is also in women, and attractive.
# 
# woman devil, distracted.

# ### premise - men/women

# In[230]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'men' == i], key=lambda x: x[1], reverse=True)[:100]


# In[231]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'women' == i], key=lambda x: x[1], reverse=True)[:100]


# Observation: man has weapons, discussions/discuss, tasks, conversing, telescopes, hell, cannon. women has mostly clothes still like bikinis, skimpy, talk, sexy, chatting, exotic.

# ### premise - girl/boy

# In[232]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'girl' == i], key=lambda x: x[1], reverse=True)[:100]


# In[233]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'boy' == i], key=lambda x: x[1], reverse=True)[:100]


# nothing interesting

# ## Race
# 
# - caucasian
# - asian
# - indian
# - chinese
# - black
# - white
# - african

# ### premise - caucasian

# In[234]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'caucasian' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - asian

# In[235]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'asian' == i], key=lambda x: x[1], reverse=True)[:100]


# In[242]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'asians' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - indian

# In[236]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'indian' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - chinese

# In[237]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'chinese' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - black

# In[239]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'black' == i], key=lambda x: x[1], reverse=True)[:1000]


# ### premise - white

# In[240]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'white' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - african

# In[241]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'african' == i], key=lambda x: x[1], reverse=True)[:100]


# In[245]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'africans' == i], key=lambda x: x[1], reverse=True)[:100]


# ## Religion
# - christian/s
# - muslim/s
# - jewish/jews
# - buddhist/s
# - sikh/s

# ### premise - christian/s

# In[246]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'christian' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - muslim/s

# In[248]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'muslim' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - jewish/jews

# In[250]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'jewish' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - buddhist/s

# In[254]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'buddhist' == i], key=lambda x: x[1], reverse=True)[:100]


# ### premise - sikh/s

# In[255]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'sikh' == i], key=lambda x: x[1], reverse=True)[:100]


# ## Age
# - young
# - old
# - elderly

# ### premise - young

# In[256]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'young' == i], key=lambda x: x[1], reverse=True)[:100]


# In[257]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'old' == i], key=lambda x: x[1], reverse=True)[:100]


# In[258]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'elderly' == i], key=lambda x: x[1], reverse=True)[:100]


# In[259]:


sorted([((i,j),z) for ((i,j),z) in pmi_premise.items() if 'attractive' == j], key=lambda x: x[1], reverse=True)[:100]


# In[400]:


sorted([((i,j),z) for ((i,j),z) in pmi_hyp.items() if 'politics' == j], key=lambda x: x[1], reverse=True)[:100]


# In[498]:


def get_pmi_for_word(word, source):
    return sorted([((i,j),z) for ((i,j),z) in source.items() if word in j], key=lambda x: x[1], reverse=True)[:100]


# In[497]:


def get_pmi_for_id(word, source):
    return sorted([((i,j),z) for ((i,j),z) in source.items() if word in i], key=lambda x: x[1], reverse=True)[:100]


# In[500]:


get_pmi_for_word('attractive', pmi_hyp)


# # ADVANCED ANALYSIS

# ## N-gram PMI

# N grams: black people, black men, black women, black woman, black man..

# In[543]:


n_grams = ['people', 'men', 'women', 'woman', 'man']
idents = ['black', 'white', 'caucasian', 'asian', 'indian', 'african', 'hispanic', 'mexican',           'christian', 'jewish', 'muslim', 'young', 'old', 'african american', 'african americans']


# In[544]:


n_gram_idents = [x + ' ' + y for x in idents for y in n_grams]
n_gram_idents += ['african american', 'african americans']


# In[545]:


n_premise_dict = {}
n_hyp_dict = {}

def fill_n_dict(source, target_dict):
    for ident in tqdm(n_gram_idents):
        for sentence in source.values:
            sentence = ' '.join(sentence[0])
            if ident in sentence:
                target_dict[ident] = target_dict.get(ident, 0) + 1
            
fill_n_dict(premises, n_premise_dict)
fill_n_dict(hypotheses, n_hyp_dict)


# In[546]:


n_premise_dict


# In[547]:


n_hyp_dict


# In[728]:


[x for x in premises[premises.index == 0].values[0][0]], train[train.index == 0]["sentence1"]


# In[731]:


train[train.index == 0]["sentence1"].values[0]


# In[739]:


co_n_premise_dict = {}
co_n_hyp_dict = {}

def fill_co_n_dict(target_dict):
    
    if target_dict == co_n_premise_dict:
        sent = "sentence1"
        source = premises
    else:
        sent = "sentence2"
        source = hypotheses
        
    for ident in tqdm(n_gram_idents):
        
        for i in source.index:
            splitted_sentence = [x for x in source[source.index == i].values]
            sentence = train[train.index == i][sent].values
            if isinstance(sentence, float):
                continue
            if ident in sentence:
                for word in splitted_sentence:
                    print(splitted_sentence, sentence)
                    break
                    if word not in ident:
                        target_dict[(ident, word)] = target_dict.get((ident, word), 0) + 1

fill_co_n_dict(co_n_premise_dict)
fill_co_n_dict(co_n_hyp_dict)


# In[549]:


co_n_premise_dict


# In[550]:


co_n_hyp_dict


# In[551]:


def sort_dict(dic):
    return sorted(dic.items(), key=lambda x: x[1], reverse=True)


# In[552]:


sort_dict(co_n_hyp_dict)


# In[553]:


def get_co_pmi(wi, wj, wi_dict, wj_dict, co_dict, N): 
    pmi = 0.0
    
    if wi not in wi_dict or wj not in wj_dict or (wi, wj) not in co_dict:
        return pmi 
    
    if wj_dict[wj] < 10:
        return pmi
    
    c_wi = wi_dict[wi]
    c_wj = wj_dict[wj]
    c_wi_wj = co_dict[(wi, wj)]
    # print(c_wi, c_wj, c_wi_wj)
    
    pmi = np.log2( (N * c_wi_wj) / (c_wi * c_wj) )
    
    return pmi


# In[554]:


pmi_conprem = {}
pmi_conhyp = {}

for idd, word in tqdm(co_n_premise_dict):
    pmi_conprem[(idd, word)] = get_co_pmi(idd, word, n_premise_dict, premise_dict, co_n_premise_dict, len(premises.values))
    
for idd, word in tqdm(co_n_hyp_dict):
    pmi_conhyp[(idd, word)] = get_co_pmi(idd, word, n_hyp_dict, hyp_dict, co_n_hyp_dict, len(hypotheses.values))


# In[555]:


sort_dict(pmi_conprem)


# In[556]:


sort_dict(pmi_conhyp)


# In[557]:


for n_gram_ident in n_gram_idents:
    get_n_highest(n_gram_ident, pmi_conprem, n=30, plot=True, rotation=45)


# In[558]:


for n_gram_ident in n_gram_idents:
    get_n_highest(n_gram_ident, pmi_conhyp, n=30, plot=True, rotation=45)


# In[542]:


get_pmi_for_word('bomb', pmi_hyp)


# # PMI GRAPHS

# **Gender**
# - male/female
# - man/woman
# - men/women
# - girl/boy
# 
# **Race**
# - caucasian
# - asian
# - indian
# - chinese
# - black
# - white
# - african
# 
# **Religion**
# - christian/s
# - muslim/s
# - jewish/jews
# - buddhist/s
# - sikh/s
# 
# **Age**
# - young
# - old
# - elderly

# In[404]:


identities_subset = ["male", "female", "man", "woman", "girl", "boy", "caucasian", "asian", "indian", "chinese", 
                     "black", "white", "african", "christian", "muslim", "jewish", "buddhist", "sikh", \
                     "young", "old", "elderly"]


# In[529]:


for identity in identities:
    get_n_highest(identity, pmi_premise, n=25, plot=True, rotation=70)


# In[530]:


for identity in identities:
    get_n_highest(identity, pmi_hyp, n=30, plot=True, rotation=70)


# In[469]:


sort_dict(p_availabilities)


# In[ ]:





# In[470]:


sort_dict(h_availabilities)


# In[518]:


find_sentence("indonesian")


# In[571]:


get_pmi_for_word('CEO', pmi_hyp)


# In[570]:


find_sentence("CEO", "", "hyp")


# In[576]:


find_sentence("terrorist", "","hyp")


# In[590]:


get_pmi_for_word("robbed", pmi_hyp)


# Problematic hypothesis

# In[594]:


find_sentence("Asian", "prostitutes", "hyp")


# In[599]:


train[train["sentence2"] == 'The Asian friends are prostitutes.']["sentence1"].values


# In[600]:


find_sentence("Muslim", "terrorists", "hyp")


# In[601]:


train[train["sentence2"] == 'The Muslims are terrorists.']["sentence1"].values


# In[603]:


find_sentence("muslim man", "weapon", "hyp")


# In[604]:


train[train["sentence2"] == 'The muslim man has a weapon.']["sentence1"].values


# In[629]:


find_sentence("black", "ghetto", "hyp")


# In[630]:


train[train["sentence2"] == 'A black male walks through the ghetto.']["sentence1"].values


# In[639]:


find_sentence("Mexican", "convicts", "hyp")


# In[640]:


train[train["sentence2"] == 'The men are escaping convicts, trying to get to the Mexican border.']["sentence1"].values


# In[645]:


find_sentence("teenagers", "stupid", "hyp")


# In[646]:


train[train["sentence2"] == 'Five teenagers are excited about something stupid.']["sentence1"].values


# In[659]:


find_sentence("women", "gossip", "hyp")


# In[665]:


train[train["sentence2"] == 'Three women gossip about their children.']["sentence1"].values


# In[694]:


find_sentence("homeless", "man", "hyp")


# In[697]:


sent = 'A boy watches a homeless man play for donations.'
train[train["sentence2"] == sent]["sentence1"].values


# In[666]:


find_sentence("woman", "attractive", "hyp")


# In[681]:


train[train["sentence2"] == 'An attractive woman is taking a stroll on a cobblestone street.']["sentence1"].values


# Problematic n-grams

# In[647]:


train[train["sentence2"] == 'The muslim man has a weapon.']["sentence1"].values


# In[653]:


find_sentence("Asian woman", "sexy", "hyp")


# In[654]:


train[train["sentence2"] == 'Asian woman at the sexy photoshoot']["sentence1"].values


# In[655]:


find_sentence("Asian man", "intoxicated", "hyp")


# In[656]:


train[train["sentence2"] == 'An Asian man is very intoxicated.']["sentence1"].values


# In[699]:


find_sentence("asian man", "bomb", "hyp")


# In[700]:


sent = 'An asian man sets the timer on a hidden bomb.'
train[train["sentence2"] == sent]["sentence1"].values


# In[712]:


find_sentence("black people", "guns")


# In[742]:


get_n_highest("black people", pmi_conprem, n=50, plot=True, rotation=45)


# In[743]:


get_n_highest("black people", pmi_conhyp, n=50, plot=True, rotation=45)


# In[741]:


get_n_highest("young man", pmi_conprem, n=50, plot=True, rotation=45)


# In[745]:


get_n_highest("young man", pmi_conhyp, n=50, plot=True, rotation=45)


# In[746]:


get_n_highest("white people", pmi_conprem, n=50, plot=True, rotation=45)


# In[747]:


get_n_highest("white people", pmi_conhyp, n=50, plot=True, rotation=45)


# In[751]:


find_sentence("people", "waist", "hyp")


# In[755]:


premise_sents = [x.lower() for x in train["sentence1"].values]
hyp_sents = [x.lower() for x in train["sentence2"].values if not isinstance(x, float)]


# In[762]:


prem_sents_id_counts = {}
hyp_sents_id_counts = {}

chosen = ["young man", "young woman", "muslim man", "asian man", "asian women", "white people", "black people"]
for ident in tqdm(n_gram_idents):
    for sent in (premise_sents):
        if ident in sent:
            prem_sents_id_counts[ident] = prem_sents_id_counts.get(ident, 0) + 1
    for sent in (hyp_sents):
        if ident in sent:
            hyp_sents_id_counts[ident] = hyp_sents_id_counts.get(ident, 0) + 1
            
    


# In[763]:


prem_sents_id_counts


# In[764]:


hyp_sents_id_counts


# In[784]:


prem_co_counts = {}
hyp_co_counts = {}

for ident in tqdm(n_gram_idents):
    for sent in (premise_sents):
        if ident in sent:
            for word in list(set(sent.replace('.', ' ').replace(',',' ').split())):
                if word not in ident and word not in identities and word not in stop_words:
                    prem_co_counts[(ident, word)] = prem_co_counts.get((ident, word), 0) + 1
    for sent in (hyp_sents):
        if ident in sent:
            for word in list(set(sent.replace('.', ' ').replace(',',' ').split())):
                if word not in ident and word not in identities and word not in stop_words:
                    hyp_co_counts[(ident, word)] = hyp_co_counts.get((ident, word), 0) + 1


# In[785]:


sort_dict(prem_co_counts)


# In[786]:


sort_dict(hyp_co_counts)


# In[ ]:


def get_co_pmi(wi, wj, wi_dict, wj_dict, co_dict, N): 
    pmi = 0.0
    
    if wi not in wi_dict or wj not in wj_dict or (wi, wj) not in co_dict:
        return pmi 
    
    if wj_dict[wj] < 10:
        return pmi
    
    c_wi = wi_dict[wi]
    c_wj = wj_dict[wj]
    c_wi_wj = co_dict[(wi, wj)]
    # print(c_wi, c_wj, c_wi_wj)
    
    pmi = np.log2( (N * c_wi_wj) / (c_wi * c_wj) )
    
    return pmi


# In[789]:


prem_copmi = {}
hyp_copmi = {}

for (idd, word) in tqdm(prem_co_counts):
    prem_copmi[(idd, word)] = get_co_pmi(idd, word, prem_sents_id_counts, premise_dict, prem_co_counts, len(premises.values))
    
for (idd, word) in tqdm(hyp_co_counts):
    hyp_copmi[(idd, word)] = get_co_pmi(idd, word, hyp_sents_id_counts, hyp_dict, hyp_co_counts, len(hypotheses.values))


# In[791]:


sort_dict(prem_copmi)


# In[793]:


sort_dict(hyp_copmi)


# In[794]:


for identity in n_gram_idents:
    get_n_highest(identity, 
                , n=25, plot=True, rotation=70)


# In[795]:


for identity in n_gram_idents:
    get_n_highest(identity, hyp_copmi, n=25, plot=True, rotation=70)


# In[876]:


C


# In[896]:


sent = list(find_sentence(" men ", "discuss", "hyp"))[3]
print(sent)
train[train["sentence2"] == sent]["sentence1"], train[train["sentence2"] == sent]["label1"]


# In[895]:


train.head()


# In[819]:


for identity in ["caucasian"]:
    get_n_highest(identity, pmi_hyp, n=30, plot=True, rotation=70)


# In[ ]:


# woman attractive
# man homeless
# women gossip
# men discuss
# asian prostitutes, knowledge
# mexican, convicts
# muslim, terrorists
# teenagers, loiter, stupid


# In[908]:


sent = list(find_sentence(" woman ", "attractive", "hyp"))[6]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[916]:


sent = list(find_sentence(" man ", "homeless", "hyp"))[10]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[921]:


sent = list(find_sentence("women", "gossip", "hyp"))[14]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[923]:


sent = list(find_sentence("men", "discuss", "hyp"))[5]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[928]:


sent = list(find_sentence("Asian", "prostitutes", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[929]:


sent = list(find_sentence("Asian", "knowledge", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[930]:


sent = list(find_sentence("Mexican", "convicts", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[974]:


sent = list(find_sentence("African", "ghetto", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[931]:


sent = list(find_sentence("Muslim", "terrorists", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[941]:


sent = list(find_sentence("teenagers", "loiter", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[933]:


sent = list(find_sentence("teenagers", "stupid", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[942]:


# black people rioting, weapon
# asian woman, sexy
# asian man, literature
# african ammerican, jaywalking
# muslim man, weapon
# young woman, scantily
# old man, creepy


# In[943]:


sent = list(find_sentence("black people", "rioting", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[944]:


sent = list(find_sentence("black people", "weapon", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[946]:


sent = list(find_sentence("Asian woman", "sexy", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[947]:


sent = list(find_sentence("Asian man", "literature", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[971]:


sent = list(find_sentence("African American", "jaywalking", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[950]:


sent = list(find_sentence("muslim man", "weapon", "hyp"))[0]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[969]:


sent = list(find_sentence("old man", "creepy", "hyp"))[2]
print("HYP: ", sent)
print("PREMISE: ", train[train["sentence2"] == sent]["sentence1"].values[0])
print("LABEL: ", train[train["sentence2"] == sent]["label1"].values[0])


# In[ ]:





# In[ ]:




