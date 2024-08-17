import pandas as pd
import numpy as np
import random
from datetime import datetime
import re 


def prepare_input_dataframe(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Prepares a dataframe to be used as input for a machine learning model by ensuring all
    expected columns are present and in the correct order.

    Parameters:
    - df: Input dataframe containing some of the expected columns.
    - expected_columns: List of all columns the model expects as input.

    Returns:
    - A dataframe with all the expected columns, filling missing ones with 0.
    """
    # Step 1: Add missing columns with a default value (e.g., 0)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Step 2: Reorder columns to match the expected order
    df = df[expected_columns]
    
    return df


def run_data_prep(df: pd.DataFrame) -> pd.DataFrame:
    df = select_only_relevant_candidates(df)
    df = clean_date_columns(df, ['cdate', 'belafspraak', 'geboortedatum'])
    df = clean_and_combine_source_columns(df)
    df = merging_two_columns(df, "utm_medium", "medium")
    df = clean_utm_campaign(df)
    df = split_campaign_locations(df)
    df = split_adgroup_locations(df)
    df = convert_pagina_to_parent_page(df)
    df = merge_UitkomstTelefonisch(df)
    df = cleaning_leeftijd(df)
    df = clean_rijbewijs(df)
    df = clean_eigen_vervoer(df)
    df = clean_score_1(df)
    df = bereken_jaar_ervaring(df)
    df = convert_postcode(df)
    df = clean_werksituatie(df)
    df['Voorkeursbranche'] = df['Voorkeursbranche'].str.lower()
    df = clean_strevon_startsalaris_werktijden(df, "Strevon startsalaris")
    df = clean_strevon_startsalaris_werktijden(df, "Strevon werktijden")
    df['diff_days'] = (df['belafspraak'] - df['cdate']).dt.days

    #prep for modelling
    df = date_cols_to_numeric(df)
    df = prep_dataset_for_modelling(df)
    return df


def prep_dataset_for_modelling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a dataset for modeling by converting categorical variables into dummy/indicator variables.
    This function identifies categorical columns (with an 'object' data type) in the input DataFrame,
    converts them into dummy variables, and then concatenates them with the non-categorical columns.
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame that contains both categorical and non-categorical data.
    Returns:
    --------
    pd.DataFrame
        A DataFrame where all categorical variables have been replaced with their dummy variables, 
        and all non-categorical columns are preserved.
    """
    # Separate categorical (object) and non-categorical columns
    categorical_columns: pd.DataFrame = df.select_dtypes(include='object')
    non_categorical_columns: pd.DataFrame = df.select_dtypes(exclude='object')

    # Dummy code categorical columns and concatenate with non-categorical columns
    df_final: pd.DataFrame = pd.concat([non_categorical_columns, pd.get_dummies(categorical_columns)], axis=1)

    return df_final


def clean_strevon_startsalaris_werktijden(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Cleans a specified column in the DataFrame by replacing values that do not match
    the allowed answers with NaN.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the column to be cleaned.
    column : str
        The name of the column to be cleaned.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the specified column cleaned, where invalid answers are replaced with NaN.
    """
    answers = ["Dit is haalbaar", "Dit is een uitdaging", "Dit is niet haalbaar"]
    df[column] = df[column].apply(lambda x: np.nan if x not in answers else x)
    return df

def clean_werksituatie(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the 'Werksituatie' column in the DataFrame by normalizing text values,
    replacing specific strings, and filtering based on valid answers.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the 'Werksituatie' column to be cleaned.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the 'Werksituatie' column cleaned and standardized.
    """
    df['Werksituatie'] = df['Werksituatie'].apply(lambda x: x.lower() if math.isnan(x) is False else x)
    df['Werksituatie'].replace("werkloos", "ik ben werkloos", inplace=True)

    klus_answers = ['niks: 0 klussen', 'weinig: 3 tot 4 klussen', 'regelmatig: 5 tot 8 klussen',
                    'bij uitzondering: 1 tot 2 klussen', 'veel: meer dan 8 klussen']
    df['Werksituatie'] = df['Werksituatie'].apply(lambda x: "ik ben zzp'er" if x in klus_answers else x)

    answers = ['ik heb een tijdelijk contract (bepaalde tijd)', 'ik ben werkloos',
               'ik heb een vast contract', "ik ben zzp'er"]
    df['Werksituatie'] = df['Werksituatie'].apply(lambda x: np.nan if x not in answers else x)
    return df

def convert_postcode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts and enriches the 'postcode' column by adding city, province, and Randstad status,
    and then removes the original 'postcode' column.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the 'postcode' column.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with additional columns for city, province, and Randstad status, 
        and with the 'postcode' column removed.
    """
    df['postcode_getal'] = df['postcode'].str[:4]

    postal_code_data = pd.read_excel('model_training/data/postcodesNL.xlsx', converters={'Postcode': str})
    df = df.merge(postal_code_data, left_on='postcode_getal', right_on='Postcode', how='left')

    randstad_list = ['Amsterdam', 'Rotterdam', 'Den Haag', 'Utrecht', 'Almere', 'Haarlem', 
                     'Amersfoort', 'Zaanstad', 'Haarlemmermeer', 'Zoetermeer', 'Leiden', 
                     'Dordrecht', 'Alphen aan den Rijn', 'Westland', 'Alkmaar', 'Delft']
    df['randstad'] = df['Gemeente'].isin(randstad_list)
    df.loc[df['Gemeente'].isna(), 'randstad'] = np.nan
    df.loc[df['postcode'] == 'Overig', 'randstad'] = False

    df.drop(['postcode'], axis=1, inplace=True)
    return df

def date_cols_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts specified date columns in the DataFrame to numeric format.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing date columns to be converted.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the specified date columns converted to numeric format.
    """
    date_cols = ['cdate', 'geboortedatum', 'belafspraak']
    for col in date_cols:
        df[col] = pd.to_numeric(df[col])
    return df

def replace_fun(str_val: str) -> str:
    """
    Replaces specific substrings within the input string based on a predefined dictionary.

    Parameters:
    -----------
    str_val : str
        The input string to be modified.

    Returns:
    --------
    str
        The modified string with specified substrings replaced.
    """
    replace_dict = {
        ',': '.', ' ': '', 'half': '0.5', 'een': '1', 'twee': '2', 'drie': '3',
        'vier': '4', 'vijf': '5', 'zes': '6', 'zeven': '7', 'acht': '8', 'negen': '9',
        'months': 'maanden', 'jaren': 'jaar'
    }
    for orig, new in replace_dict.items():
        str_val = str_val.replace(orig, new)
    return str_val



def extract_vals(str_val: str) -> float:
    """
    Extracts and converts numerical values from a string, accounting for various formats.

    Parameters:
    -----------
    str_val : str
        The input string from which to extract a numerical value.

    Returns:
    --------
    float
        The extracted numerical value, or NaN if extraction is not possible.
    """
    try:
        val = float(str_val)
        return val
    except ValueError:
        if str_val in ['niet', 'geen']:
            return 0.0
        elif 'sinds' in str_val or 'vanaf' in str_val:
            matches = re.findall(r'\d{4}', str_val)
            return float(matches[0]) if matches else np.nan
        
        if 'jaar' in str_val:
            pattern = r'(\d+\.\d+|\d+)\s*jaar'
        elif 'maand' in str_val:
            pattern = r'(\d+\.\d+|\d+)\s*maand'
        else:
            return np.nan

        matches = re.findall(pattern, str_val)
        if not matches:
            return np.nan

        val = float(matches[0])
        if 'maand' in str_val:
            val /= 12
        
        return val

def calculate_years_worked(row: pd.Series) -> float:
    """
    Calculates the number of years worked based on the start year of experience and the end date.

    Parameters:
    -----------
    row : pd.Series
        A row from a DataFrame containing 'jaar_ervaring' (start year) and 'cdate' (end date).

    Returns:
    --------
    float
        The number of years worked, adjusted for partial years.
    """
    start_year = row['jaar_ervaring']
    end_date = row['cdate']

    if not isinstance(end_date, datetime):
        end_date = datetime(2024, 1, 1)

    year = int(start_year)
    fraction = start_year - year
    years_worked = end_date.year - year

    if fraction > 0:
        start_date = datetime(year, 1, 1) + pd.DateOffset(months=int(fraction * 12))
        if end_date < start_date:
            years_worked -= 1
        elif end_date > start_date:
            extra_months = (end_date - start_date).days / 365.25
            years_worked += extra_months

    return years_worked


def bereken_jaar_ervaring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the year of experience for each row in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the column 'Hoe lang in dienst/werkloos'.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with an additional 'jaar_ervaring' column containing the calculated years of experience.
    """
    df.loc[df['Hoe lang in dienst/werkloos'] == 'Vijf een half jaar', 'Hoe lang in dienst/werkloos'] = '5.5jaar'
    df['jaar_ervaring'] = df['Hoe lang in dienst/werkloos'].apply(
        lambda x: extract_vals(replace_fun(x.lower())) if isinstance(x, str) else x
    )
    df.loc[df['jaar_ervaring'] > 1950, 'jaar_ervaring'] = df[df['jaar_ervaring'] > 1950].apply(calculate_years_worked, axis=1)
    return df


def clean_score_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the 'score 1' column by replacing invalid entries with NaN.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the 'score 1' column.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the 'score 1' column cleaned.
    """
    column = 'score 1'
    answer_options = [
        "Laminaat leggen, lampen aansluiten op elektra, sleutelen aan scooter/auto",
        "Gordijnen ophangen, batterij vervangen rookmelder, lampje verwisselen",
        "Meer dan een half jaar ervaring in een technische functie (bijvoorbeeld monteur)"
    ]
    df[column] = df[column].apply(lambda x: x if x in answer_options else np.nan)
    return df

import math

def clean_eigen_vervoer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the 'beschikking tot eigen vervoer?' column.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the 'beschikking tot eigen vervoer?' column.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the 'beschikking tot eigen vervoer?' column cleaned and standardized.
    """
    column = "beschikking tot eigen vervoer?"
    df[column] = df[column].apply(lambda x: x.lower() if math.isnan(x) is False else x)
    df[column] = df[column].apply(lambda x: "nee dit heb ik niet" if str(x) in ["geen auto", "geen vervoer"] else x)
    df[column] = df[column].apply(lambda x: "ja een eigen auto of motor" if "motor" in str(x) else x)
    df[column] = df[column].apply(lambda x: "ja een eigen auto of motor" if "auto" in str(x) else x)
    df[column] = df[column].apply(lambda x: "ja een eigen auto of motor" if str(x) == "ja" else x)
    df[column] = df[column].apply(lambda x: "nee dit heb ik niet" if str(x) == "nee" else x)
    return df


def clean_rijbewijs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the 'Ben je in het bezit van rijbewijs?' column by converting text to lowercase and removing trailing commas.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the 'Ben je in het bezit van rijbewijs?' column.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the 'Ben je in het bezit van rijbewijs?' column cleaned.
    """
    column = "Ben je in het bezit van rijbewijs?"
    df[column] = df[column].apply(lambda x: x.lower().strip(",") if isinstance(x, str) else x)
    return df

def convert_pagina_to_parent_page(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'pagina' column to a 'parent_page' by extracting the first part of the path.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the 'pagina' column.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with a new 'parent_page' column derived from the 'pagina' column.
    """
    column = 'pagina'
    df[column] = df[column].apply(lambda x: x.strip("/") if isinstance(x, str) else x)
    df['parent_page'] = df[column].apply(lambda x: x.split("/")[0] if isinstance(x, str) else x)
    return df

def merge_UitkomstTelefonisch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the 'uitkomstTelefonischDeal' and 'uitkomstTelefonischContact' columns,
    prioritizing 'uitkomstTelefonischDeal', and then drops the 'uitkomstTelefonischDeal' column.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing 'uitkomstTelefonischDeal' and 'uitkomstTelefonischContact' columns.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the 'uitkomstTelefonischDeal' column merged and removed.
    """
    df['uitkomstTelefonischDeal'] = df['uitkomstTelefonischDeal'].fillna(df["uitkomstTelefonischContact"])
    df.drop("uitkomstTelefonischDeal", axis=1, inplace=True)
    return df


def categorize(x: float) -> str:
    """
    Categorizes a given numerical value into predefined age groups.

    Parameters:
    -----------
    x : float
        The age value to be categorized.

    Returns:
    --------
    str
        A string representing the age category.
    """
    if 0 < x < 31:
        return "18-30 jaar"
    elif 30 < x < 50:
        return "31-49 jaar"
    elif x >= 50:
        return "50 jaar of ouder"
    else:
        return np.nan

def cleaning_leeftijd(df):
    df['leeftijd'] = df['leeftijd'].apply(lambda x: str(random.choice(range(18,30))) if x=="18-30 jaar" else x)
    df['leeftijd'] = df['leeftijd'].apply(lambda x: str(random.choice(range(31,49))) if x=="31-49 jaar" else x)
    df['leeftijd'] = df['leeftijd'].apply(lambda x: str(random.choice(range(31,45))) if x=="31-45 jaar" else x)
    df['leeftijd'] = df['leeftijd'].apply(lambda x: str(random.choice(range(50,65))) if x=="50 jaar of ouder" else x)
    df['leeftijd'] = df['leeftijd'].apply(float)
    df['leeftijd'] = df['leeftijd'].replace(0,np.nan)
    df['leeftijd'] = df['leeftijd'].replace(1,np.nan)

    df['leeftijd_cat'] = df['leeftijd'].apply(lambda x: categorize(x))
    return df



def split_adgroup_locations(df):
    #create location of utm_adgroup variable
    column_name = "utm_adgroup"
    df["utm_adgroup_location"] = df[column_name].apply(lambda x: str(x).split("_")[-1] if len(str(x).split("_")) > 0 else None)
    locations=['amsterdam','utrecht','dordrecht','leiden','gouda','denhaag','utrecht','rotterdam']
    df["utm_adgroup_location"] = df["utm_adgroup_location"].apply(lambda x: x if x in locations else None)
    #remove locations from utm_campaigns
    df[column_name+"_no_loc"] = df[column_name].apply(lambda x: "_".join(str(x).split("_")[:-1]) if str(x).split("_")[-1] in locations else x)
    df.drop(['adgroup'],axis=1, inplace=True)
    return df


def split_campaign_locations(df):
    #create location of utm_campaign variable
    df["utm_campaign_location"] = df["utm_campaign"].apply(lambda x: str(x).split("_")[-1] if len(str(x).split("_")) > 0 else None)
    locations=['amsterdam','utrecht','dordrecht','leiden','gouda','denhaag','utrecht','rotterdam']
    df["utm_campaign_location"] = df["utm_campaign_location"].apply(lambda x: x if x in locations else None)
    #remove locations from utm_campaigns
    df["utm_campaign_no_loc"] = df["utm_campaign"].apply(lambda x: "_".join(str(x).split("_")[:-1]) if str(x).split("_")[-1] in locations else x)
    # CampagneNaam is same as utm_campaign so drop
    df.drop(['campagneNaam'], axis=1, inplace=True)
    return df

def clean_utm_campaign(df):
    # cleaning utm_campaign
    column_name = 'utm_campaign'
    df[column_name].replace('installatiemonteur_amstedram', 'installatiemonteur_amsterdam', inplace=True) #Mag dit is dit hetzelfde?
    df[column_name].replace('installatiemonteur_amsterdan', 'installatiemonteur_amsterdam', inplace=True) #Mag dit is dit hetzelfde?
    
    installatiemonteur_values = [val for val in df['utm_campaign'].unique() if "installatiemonteur-" in str(val)]
    if "installatiemonteur-exp" in installatiemonteur_values:
        installatiemonteur_values.remove('installatiemonteur-exp')
    for val in installatiemonteur_values:
        df[column_name].replace(val, 'installatiemonteur', inplace=True)

    elektromonteur_values = [val for val in df['utm_campaign'].unique() if "elektromonteur-" in str(val)]
    elektromonteur_values += ["elektromonteur50215728-102-1"]
    for val in elektromonteur_values:
        df[column_name].replace(val, 'elektromonteur', inplace=True)

    installatietechniek_values = [val for val in df['utm_campaign'].unique() if "installatietechniek-" in str(val).lower()]
    for val in installatietechniek_values:
        df[column_name].replace(val, 'installatietechniek', inplace=True)

    elektrotechniek_values = [val for val in df['utm_campaign'].unique() if "elektrotechniek-" in str(val).lower()]
    for val in elektrotechniek_values:
        df[column_name].replace(val, 'elektrotechniek', inplace=True)
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

    df['utm_source'] = df['utm_source'].str.lower()
    df["utm_source"].replace('werkenbijstrevon', 'strevon',inplace=True) #Mag dit is dit hetzelfde?
    df['source'] = df['source'].replace("direct","strevon")
    df['utm_source'] = df['utm_source'].replace("direct","strevon")
    
    merging_two_columns(df, 'utm_source', 'source')
    return df