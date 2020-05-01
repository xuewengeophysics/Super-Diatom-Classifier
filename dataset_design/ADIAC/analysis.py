# %%
import cv2
import pandas as pd


# %%
df_adiac = pd.read_csv("/mnt/1184aa73-c854-40a5-9a6e-30ae55a1cbf8/MT/ADIAC_dataset/index.csv")
#df_adiac = df_adiac[df_adiac["image_type"]=='single valve, testset2']
df_2011 = pd.read_csv("/mnt/1184aa73-c854-40a5-9a6e-30ae55a1cbf8/MT/ADIAC_dataset/2011_ref.csv")
print(df.shape)
print(df.head())

#%%
df_38 = df_2011[df_2011["38p"].isnull()==False]
df_48 = df_2011[df_2011["48p"].isnull()==False]
df_55 = df_2011[df_2011["55p"].isnull()==False]
print(df_38.shape)
print(df_48.shape)
print(df_55.shape)

# %%
filter_df = df_55
def get_dict(df):
    kept_genus = {}
    for index, row in df.iterrows():
        split_taxon = row['taxon'].split("/")
        if len(split_taxon)==2:
            kept_genus[split_taxon[0]].append(split_taxon[1])
        else:
            kept_genus.setdefault(split_taxon[0], [])
    return kept_genus
dict38 = get_dict(df_38)
dict48 = get_dict(df_48)
dict55 = get_dict(df_55)

# %%
print(df["processing"].unique())
print(df["qualifier"].unique())
print(df["image_type"].unique())
df = df[df["processing"]!="scale bar added"]

# %% TEST_SET
genus_pattern = '|'.join(kept_genus)
species_pattern = '|'.join(kept_species)
test_set2 = df[df["image_type"]=='single valve, testset2']
test_set2 = test_set2[test_set2["genus"].str.contains(genus_pattern)]
test_set2 = test_set2[test_set2["species"].str.contains(species_pattern)]
print(test_set2.shape)
test_set2 = test_set2.filter(["image", "genus", "species"])
test_set2.to_csv("38taxa.csv")
test_set2.head()
# %%
genus = {}
for index, row in test_set2.iterrows():
    genre = row['genus']
    genus.setdefault(genre, 0)
    genus[genre]+=1
print(len(list(genus.keys())))

#%%
df_filtered
# %%
Cocconeis/neodiminuta
