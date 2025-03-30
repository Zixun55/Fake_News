import pandas as pd

#%%
# Combine multiple explanations in Politifact with "|"
file_name = 'politifact_true'
df = pd.read_csv('./knowledge_base/'+file_name+'.csv')
df['Explanation'] = df['Explanation'].str.replace('\n', '|')

df.to_csv('./knowledge_base/clean_data/'+file_name+'_clean.csv', index=False)

#%%
# Count each classification in MBFC
df = pd.read_csv('./knowledge_base/mbfc.csv')
classification = df['Classification'].unique()

true = len(df[df.Classification == 'TRUE'])
mostly_true = len(df[df.Classification == 'MOSTLY TRUE'])

false = len(df[df.Classification == 'FALSE'])
mostly_false = len(df[df.Classification == 'MOSTLY FALSE'])
half_true = len(df[df.Classification == 'HALF TRUE'])
misleading = len(df[df.Classification == 'MISLEADING'])
blatant_lie = len(df[df.Classification == 'BLATANT LIE'])


print(classification, '\n')
print('TRUE', true)
print('MOSTLY TRUE', mostly_true)

print('FALSE', false)
print('MOSTLY FALSE', mostly_false)
print('HALF TRUE', half_true)
print('BLATANT LIE', blatant_lie)
print('MISLEADING', misleading)

#%%
# politifact_{}_clean: the news with 'explanation'
df = pd.read_csv('./knowledge_base/clean_data/politifact_pants_fire_clean.csv')
politifact = df[df.Explanation != 'No Explanation available']
politifact.to_csv('./politifact.csv', index=False)

#%%
# Remove 'Post' and 'Source' for retriever
import pandas as pd
df = pd.read_csv('./knowledge_base/clean_data/politifact_all_original.csv')
df = df.drop(columns=['Post', 'Source'])
df.to_csv('./knowledge_base/clean_data/politifact_all.csv', index=False)

#%%
# Split 'Claim' and 'Explanation' for retriever
import pandas as pd
df = pd.read_csv('./knowledge_base/clean_data/politifact_all.csv')
df = df.drop(columns=['Date', 'Classification'])

df_claim = df[['Claim']]  
df_explanation = df[['Explanation']]
df_claim_explanation = df[['Claim', 'Explanation']]  

df_claim.to_csv('./retriever/politifact_all_claim.csv', index=False)  
df_explanation.to_csv('./retriever/politifact_all_explanation.csv', index=False)  
df_claim_explanation.to_csv('./retriever/politifact_all_claim_explanation.csv', index=False)
# %%
import pandas as pd

df = pd.read_csv('./classification_fake6000.csv')
classification = df['Classification'].unique()

true = (df['Classification'] == True).sum()
false = (df['Classification'] == False).sum()
nan_count = df['Classification'].isna().sum()

print(classification, '\n')
print('TRUE: ', true)
print('FALSE: ', false)
print('NaN:', nan_count)

# %%
