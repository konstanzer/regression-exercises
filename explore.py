import seaborn as sns
import matplotlib.pyplot as plt


def plot_variable_pairs(df):
    """
    Accepts a dataframe as input and plots all of the pairwise
    relationships along with the regression line for each pair
    """
    sns.pairplot(df)
    plt.show()

def months_to_years(df):
    """
    Accepts your telco churn dataframe and returns a dataframe
    with a new feature tenure_years, in complete years as a customer.
    """
    df['tenure_years'] = df.tenure/12
    return df

def plot_categorical_and_continuous_vars(df, x1, x2):
    """
    Accepts your dataframe and the name of the columns that hold the 
    continuous and categorical features and outputs 3 different
    plots for visualizing a categorical variable and a continuous variable.
    """
    
    sns.relplot(x=x1, y=x2, data=df)
    plt.show()
    
    sns.lmplot(x=x1, y=x2, data=df, line_kws={'color': 'red'})
    plt.show()
    
    sns.jointplot(x=x1, y=x2, data=df,  kind='reg', height=5)
    plt.show()
    
    