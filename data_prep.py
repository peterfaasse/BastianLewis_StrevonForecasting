import pandas as pd

def run_data_prep(df: pd.DataFrame) -> pd.DataFrame:
    df = select_only_relevant_candidates(df)
    df = clean_date_columns(df, ['cdate', 'belafspraak', 'geboortedatum'])
    df = clean_and_combine_source_columns(df)
    df = merging_two_columns(df, "utm_medium", "medium")

    return df

def select_only_relevant_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ### Function removes irrelevant candidats from list
    if "prioriteit" in df.columns.to_list():
        len_df = len(df)
        df = df[df['prioriteit'] == 1]
        df.drop(labels = ['prioriteit'],  axis= 1, inplace = True)
        print(f"removed {len_df - len(df)} candidates who did not have prioriteit 1")
    if "afwijsBasisGegevens" in df.columns.to_list():
        len_df = len(df)
        df = df[pd.isnull(df['afwijsBasisGegevens'])]
        df.drop(labels = ['afwijsBasisGegevens'],  axis= 1, inplace = True)
        print(f"removed {len_df - len(df)} candidates who did not have NULL for afwijsBasisGegevens")
    return df

# convert date strings to datetime
def clean_date_columns(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    for date_col in date_columns: 
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
    return df


# Cleaning values with low frequency <10
def clean_categorical_variable(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    print(f"working on {column_name}")
    print(f"{len(df[df[column_name].isna()])} missings")
    print("recategorizing all values that appear less then 10 times to 'other'")

    value_counts = df[column_name].value_counts()
    values_to_replace = value_counts[value_counts < 10].index
    values_to_keep = value_counts[value_counts >= 10].index
    df[column_name] = df[column_name].apply(lambda x: 'other' if x in values_to_replace else x)
    print(f"replaced the following values with other: {values_to_replace}")
    print(f"Keeping the following values: {values_to_keep}")
    print("_________________________")
    return df


def merging_two_columns(df, column_a, column_b):
    # Visualizeing, Cleaning and merging utm_medium and medium
    df[column_b] = df[column_b].str.lower()
    df[column_b] = df[column_b].replace("onbekend", "undefined")

    df[column_a] = df[column_a].str.lower()
    df[column_a] = df[column_a].replace("onbekend","undefined")
    df[column_a] = df[column_a].fillna(df[column_b])

    x = "undefined"
    df[column_a] = df.apply(lambda row: row[column_b] if row[column_a] == x else row[column_a],
                            axis=1)

    df.drop(column_b, axis=1, inplace = True)
    return df


def clean_and_combine_source_columns(df):
    df['source'] = df['source'].str.lower()
    df['source'] = df['source'].replace("nationale beroepengids", "nationaleberoepengids")
    df['source'] = df['source'].replace("werkenbijstrevon", "strevon")
    clean_categorical_variable(df, 'source')

    df['utm_source'] = df['utm_source'].str.lower()
    df["utm_source"].replace('werkenbijstrevon', 'strevon',inplace=True) #Mag dit is dit hetzelfde?
    df['source'] = df['source'].replace("direct","strevon")
    df['utm_source'] = df['utm_source'].replace("direct","strevon")
    clean_categorical_variable(df, 'utm_source')
    
    merging_two_columns(df, 'utm_source', 'source')
    return df