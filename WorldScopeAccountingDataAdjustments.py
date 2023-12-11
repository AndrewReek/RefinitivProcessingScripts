import pandas as pd


### Fiscal Year Date Adjustment - Before or in Month of June
def fiscal_year_end_adjustment(df, fy_month_col_name, query_year_col_name, adj_year_col_name):
    
    for year in df[query_year_col_name].unique():

        ### Adjustment - FY End Date in or before June
        df.loc[(df[query_year_col_name] == year) & (df[fy_month_col_name] <= 6), adj_year_col_name] = year - 1

        ### Adjustment - FY End Date after June
        df.loc[(df[query_year_col_name] == year) & (df[fy_month_col_name] > 6), adj_year_col_name] = year

    return df


### Dataset Cleaning
def implement_dataframe_cleaning(df, firm_id_col, year_col, numerical_force_cols, non_miss_cols, na_values, fill_missing_dict, dtype_dict):
    # fix numerical datatypes
    df = df.replace(na_values, '')
    for col in numerical_force_cols:
        df[col] = pd.to_numeric(df[col])

    # fill missing values
    df = df.fillna(fill_missing_dict)

    # missing value filter
    df = df.dropna(subset = non_miss_cols)

    # dropping duplicate values
    df = df.drop_duplicates(subset = [firm_id_col, year_col])

    # fix datatypes
    df = df.astype(dtype_dict)

    return df

### Filter Dataset
def implement_sample_filters(df, positive_columns, industry_col, excluded_industries):
    # filter for insolvency
    for col in positive_columns:
        df = df.loc[df[col] > 0]

    # filter for financial firms
    df = df.loc[~df[industry_col].isin(excluded_industries)]

    return df

### Industry Classification
def industry_classifier(sic_code):
    
    if sic_code >= 100 and sic_code <= 999:
        classification = "Agriculture, Fishing, Forestry"
        
    elif sic_code >= 1000 and sic_code <= 1799:
        classification = "Mining and Construction"
        
    elif sic_code >= 2000 and sic_code <= 2999:
        classification = "Light Manufacturing"
        
    elif sic_code >= 3000 and sic_code <= 3999:
        classification = "Heavy Manufacturing"
        
    elif sic_code >= 4000 and sic_code <= 4999:
        classification = "Transportation and Public Utility"
        
    elif sic_code >= 5000 and sic_code <= 5999:
        classification = "Wholesale and Retail"
        
    elif sic_code >= 6000 and sic_code <= 6999:
        classification = "Finance, Insurance, Real Estate"
        
    elif sic_code >= 7000 and sic_code <= 8999:
        classification = "Services"
        
    elif sic_code >= 9100 and sic_code <= 9729:
        classification = "Public Administration"
        
    else:
        classification = "Non-classifiable"
    
    return classification

### Winsorization Function
def winsorization(df, variables, lower_thresh = 0.01, upper_thresh = 0.99, by = 'year', suffix = '_w'):
    dfs = []

    categories = df[by].unique()

    for cat in categories:
        df_cat = df.loc[df[by] == cat]

        for var in variables:
            df_cat[f'{var}{suffix}'] = df_cat[var].clip(lower = df_cat[var].dropna().quantile(lower_thresh),
                                                        upper = df_cat[var].dropna().quantile(upper_thresh))
        dfs.append(df_cat)

    df_output = pd.concat(dfs)
    
    return df_output