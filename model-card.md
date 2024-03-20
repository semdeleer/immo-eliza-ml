# Model Card: BaggingRegressor for House Price Prediction

## Overview

- **Model Name**: BaggingRegressor
- **Model Version**: 0.15.
- **Date**: 20/03/2024
- **Authors**: Sem Deleersnijder
- **Contact**: [Sem Deleersnijder](https://github.com/semdeleer)

## Model Details

- **Framework/Libraries**: scikit-learn
- **Programming Language**: Python
- **Model Input**: 
  - `postal_code`: 
  - `property_type`: 
  - `property_subtype`: 
  - `type_of_sale`: 
  - `living_area`: 
  - `kitchen_type`: 
  - `fully_equipped_kitchen`: 
  - `open_fire`: 
  - `terrace`: 
  - `terrace_area`: 
  - `garden`: 
  - `garden_area`: 
  - `surface_of_good`: 
  - `number_of_facades`: 
  - `swimming_pool`: 
  - `state_of_building`: 
  - `main_city`: 
  - `province`:
- **Model Output**: Predicted price of the house

## Intended Use

- **Primary Use Case**: Predicting house prices based on property features.
- **Intended Users**: Real estate agents, homeowners, property investors.
- **Context of Use**: The model is intended to be used for estimating the market value of houses in Belgium, particularly in the West Flanders province.

## Evaluation

- **Metrics**: Mean Absolute Error, Mean Squared Error, R-squared
- **Datasets**: Historical real estate transaction data from Belgium.
- **Performance**: The model achieved a Mean Absolute Error of X, Mean Squared Error of Y, and R-squared of Z on the test dataset.

## Ethical Considerations

- **Bias**: Bias may exist if the training data disproportionately represents certain demographics or regions.
- **Fairness**: Efforts have been made to ensure fairness in predictions across different property types and locations.
- **Privacy**: The model does not handle sensitive personal information directly.

## Limitations

- **Data Limitations**: Limited availability of comprehensive real estate data may affect the accuracy of predictions.
- **Algorithm Limitations**: BaggingRegressor may struggle with capturing complex non-linear relationships in the data.
- **Contextual Limitations**: Market fluctuations and external economic factors may impact the model's accuracy over time.

## Future Work

- **Improvements**: Exploring advanced feature engineering techniques to capture more nuanced property characteristics.
- **Extensions**: Integrating additional external data sources such as economic indicators and demographic information for better prediction accuracy.
