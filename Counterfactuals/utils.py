import numpy as np
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Feature:
    def __init__(self, name, initial_value, boundaries, boundary_type):
        self.name = name
        self.initial_value = initial_value
        self.boundaries = boundaries
        self.boundary_type = boundary_type

        self.value = None
        self.value_history = []

    def sample(self, trial): 
        """
        Sample a value for the feature within the given boundaries. 
        """
        
        if self.boundary_type == "range":
            self.value = trial.suggest_float(self.name, self.boundaries[0], self.boundaries[1])
            self.value_history.append(self.value)

        else:
            # Fix categorical warning by converting to int 
            self.boundaries = [int(b) if isinstance(b, np.integer) else b for b in self.boundaries] 
            self.value = trial.suggest_categorical(self.name, self.boundaries)
            self.value_history.append(self.value)



def generate_individual(X, x, feature_info):
    """
    Returns a list of Feature objects, one per feature, containing:
    - feature name (as per dataframe)
    - value (actual value of feature in data point)
    - boundary (range of possible feature values for the counterfactual)
    - boundary type (whether feature value can increase/decrease, is categorical, etc)
    """

    features = []

    for _i, _info in enumerate(feature_info):
        value = x.values[0][_i] # Feature value of x
        feature_name, boundary_type = _info
        
        # Features with boundary type 'unique' can only take values from a given set
        if boundary_type == 'unique':
            boundary = X[feature_name].unique() 

        # Features with boundary type 'increase' can only take integer values 
        # above (and including) the current value from x
        elif boundary_type == 'increase': 
            lower_boundary = value
            upper_boundary = X[feature_name].max()
            boundary = (lower_boundary, upper_boundary)

        # Features with boundary type 'fixed' can only take one given value 
        elif boundary_type == 'fixed':
            boundary = (value, value)

        # Features with boundary type 'range' can take values from a specified 
        # range around the given value from x 
        elif boundary_type == 'range':
            lower_boundary = X[feature_name].min()
            upper_boundary = X[feature_name].max()
            boundary = (lower_boundary, upper_boundary)

        features.append(Feature(feature_name, value, boundary, boundary_type)) 
    
    return features

def load_data(data_filepath):
    """
    Load the LoanApprovalPrediction.csv data provided in the course
    and fill custom made dictionary containing necessary feature info
    for counterfactual search
    """

    df = pd.read_csv(data_filepath)
    df = df.drop('Loan_ID', axis=1)
    df = df.dropna()

    feature_config = {
        "categorical": ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"],

        "feature_info": [
            ('Gender', 'fixed'),
            ('Married', 'unique'),
            ('Dependents', 'fixed'),
            ('Education', 'increase'),
            ('Self_Employed', 'unique'),
            ('ApplicantIncome', 'range'),
            ('CoapplicantIncome', 'range'),
            ('LoanAmount', 'fixed'),
            ('Loan_Amount_Term', 'unique'),
            ('Credit_History', 'unique'),
            ('Property_Area', 'unique'),
        ],

        "categorical_features": ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
    }

    return df, feature_config


def encode_features(df, categorical):
    """
    Input:
        df: dataframe containing feature names in columns
        caterogiral: list of all categorical feature names.

    Custom encoding function,
    Education is manually encoded to ensure correct boundaries when constructing counterfactuals 
    The remaining categorical features are encoded using LabelEncoder

    Returns:
        modified dataframe
    """

    print("--- \nEncoded categorical features as follows:")
    label_encoders = {}

    for _col in categorical:
        if _col == "Education":
            mapping = {"Not Graduate": 0, "Graduate": 1}
            df[_col] = df[_col].map(mapping)
            label_encoders[_col] = mapping  
        else:
            le = LabelEncoder()
            df[_col] = le.fit_transform(df[_col])
            label_encoders[_col] = le
            
    for _col in categorical: 
        if _col == "Education":
            print(_col, ": ", label_encoders[_col])
        else:
            classes = label_encoders[_col].classes_
            label_mapping = {label: idx for idx, label in enumerate(classes)}
            print(_col, ": ", label_mapping)

    print('---')
    return df

def get_train_test_data(df, test_size=0.3):
    """
    Input:
        pandas dataframe
    Returns:
        random split into train and test data
    """

    data_features = df.columns

    target = data_features[-1]
    data_features = data_features[:-1]

    X = df[data_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """"
    Input:
        training data and labels
    Returns:
        XGBoost classifier model object with default parameters
    """

    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Input:
        model, test data and targets
    Shows a histogram with targets and model predictions

    Returns:
        None
    """
    
    probs = model.predict_proba(X_test)[:,1]
    plt.hist(y_test, color="green", alpha=0.5, label="targets")
    plt.hist(probs, color="blue", alpha=0.5, label="predictions")
    plt.legend()
    plt.show()
    return


def epsilon_rounding(x, x_prime, epsilon):
    """
    If very small float changes, revert to original value.
    """
    for feature in x.columns.values:
        if type(x[feature].values[0]) == np.float64:
            if np.abs(x[feature].values[0] - x_prime[feature].values[0]) <= epsilon:
                x_prime[feature] = x[feature].values[0]


def get_relevant_candidates(study, x, model, y_target, tol): 
    """
    Return relevant counterfactuals from optimisation. 
    """

    y_primes = []
    candidates = []

    # Iterate over best trials and update feature values in x_prime 
    for trial in study.best_trials:  
        x_prime = x.copy()
        for param in x.columns.values:
            if param in trial.params: # Iterate over features 
                x_prime[param] = trial.params[param] 
        epsilon_rounding(x, x_prime, 1e-1)

        # predict with model
        prediction = model.predict_proba(x_prime)[0][1]
        y_primes.append(prediction)
        candidates.append(x_prime.values[0])

    y_primes = np.array(y_primes)
    candidates = np.array(candidates)

    # Check if any counterfactual candidates meet the tolerance condition and select candidates 
    eps_condition = np.abs(y_primes - y_target) <= tol
    relevant_candidates = candidates[eps_condition]

    relevant_candidates = pd.DataFrame(relevant_candidates, columns=x.columns)
    relevant_candidates = relevant_candidates.drop_duplicates().reset_index(drop=True) # Remove duplicates and reset index 

    return relevant_candidates
