# Best-crop-selection
Using machine learning to help farmers select the best crops


In this project, I apply machine learning to build a multi-class classification model to predict the type of "crop", while using techniques to avoid multicollinearity.


Measuring essential soil metrics such as nitrogen, phosphorus, potassium levels, and pH value is crucial for assessing soil condition. However, these measurements can be expensive and time-consuming, often forcing farmers to prioritize certain metrics based on budget constraints.

Farmers have various options when deciding which crop to plant each season, with the primary objective of maximizing crop yield. One critical factor influencing crop growth is the soil condition, which can be assessed by measuring key elements such as nitrogen and potassium levels. Each crop has specific soil conditions that are ideal for optimal growth and maximum yield.

The provided dataset named soil_measures.csv, includes the following columns:

"N": Nitrogen content ratio in the soil
"P": Phosphorus content ratio in the soil
"K": Potassium content ratio in the soil
"pH": pH value of the soil
"crop": Categorical values representing various crops (target variable)

Each row in this dataset represents the soil measurements of a particular field, with the "crop" column indicating the optimal crop choice based on these measurements. I leverage this data to assist the farmer in making informed decisions about which crop to plant, ensuring the best possible yield given the soil conditions.
