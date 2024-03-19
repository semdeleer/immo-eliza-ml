# immo-eliza-ml
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
![pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![vsCode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![seaborn](https://img.shields.io/badge/Seaborn-007ACC?style=for-the-badge&logo=seaborn&logoColor=white)
)


## 📖 Description
This project is a follow up on the Immoweb EDA project in https://github.com/NathNacht/immo-eliza-scraping-immozila-Cleaning-EDA.git.

Our aim for this project is to create a machine learning model capable of predicting housing prices accurately. Our task involves cleaning and analyzing scraped data efficiently to uncover insights. Our main goal is to estimate property values and identify the most valuable ones.

To start, we've divided the raw data into two datasets: one for houses and another for apartments.
The fields within these raw files remain consistent, consisting of:


* property_id
* locality_name
* postal_code
* street_name
* house_number
* latitude
* longitude
* property_type (house or apartment)
* property_subtype (bungalow, chalet, mansion, ...)
* price
* type_of_sale (note: exclude life sales)
* number_of_rooms (Number of rooms)
* living_area (Living area (area in m²))
* kitchen_type
* fully_equipped_kitchen (0/1)
* furnished (0/1)
* open_fire (0/1)
* terrace
* terrace_area (area in m² or null if no terrace)
* garden
* garden_area (area in m² or null if no garden)
* surface_of_good
* number_of_facades
* swimming_pool (0/1)
* state_of_building (new, to be renovated, ...)


## 🛠 Installation

* clone the repo
```bash
git git@github.com:semdeleer/immo-eliza-ml.git
```

* Install all the libraries in requirements.txt
```bash
pip install -r requirements.txt
```

* Run the script
```bash
$ python3 predict.py
```

* Enter the criteria your house has

* The model will give a price back

## 👾 Cleaning steps

Before utilizing the raw data in the machine learning model, it undergoes several preprocessing steps to enhance its quality and suitability for analysis. The following are the sequential cleaning steps employed:

* Replace Outliers: Identify and substitute outliers with appropriate values.

* Drop Columns: Remove irrelevant or redundant columns from the dataset.

* Drop Null Values: Eliminate rows containing missing data.

* Transform Categorical Variables (One-Hot Encoding): Convert categorical variables into numerical format using One-Hot Encoding.

* Transform Label Encoder: Encode categorical variables using LabelEncoder.

* Create X and y: Separate the dataset into feature matrix (X) and target variable (y).

* Replace NaN in Categorical Variables: Replace missing values in categorical variables with suitable representations.

* Change to Integer: Convert data types to integer where appropriate.

* Replace Null Values with Median: Substitute missing numerical values with the median of respective columns.

* Replace Null Values with Minimum minus One: Replace missing numerical values with (minimum value - 1) of respective columns.

* Replace Null Values with Mean: Fill missing numerical values with the mean of respective columns.

* Replace Null Values with Zero: Replace missing numerical values with zero.


## 🤖 Project File structure
```
├── cleaning
│   ├── preproccessingPipeline.ipynb
├── data
│   ├── cleaned
│   │   
│   └── raw
│       └── clean_app.csv
│       └── clean_house.csv
│       └── properties.csv
├── train
│   └── train.py
├── model_pickle
├── predict.py
├── README.md
└── requirements.txt
```


## 🔍 Contributors
- [Sem Deleersnijder](https://github.com/semdeleer)

## 📜 Timeline

This project was created in 5 days.