def clean_data(df):
    df.drop_duplicates(inplace = True)
    df.drop(columns = ['species_id','measurement_id'], inplace = True)
    df.rename(columns = {"species_name": "species"}, inplace = True)
    dummy_df = pd.get_dummies(df[['species']], drop_first = True)
    return pd.concat([df, dummy_df], axis=1)