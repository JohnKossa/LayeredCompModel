# Goal

The goal of the Layered Comp Model is to create a hierarchical prediction system that starts with a general predicted price and refines it by adding information and narrowing the comparison group. This is achieved by building a tree of nodes, each representing a filtered subset of the data, and calculating a weighted average of the Wilson means to produce a final prediction. The model aims to balance accuracy and equity by penalizing large sets and promoting normativity in predictions. The implementation should be a scikit-learn compatible estimator.

More details can be found in MODEL_SPEC.md

# Tech Stack

* pandas
* python
* numpy
* scipy
* scikit-learn

# Key Processes

## Training

### Accepts
A pandas dataframe

A target prediction field

A list of columns to use

A list of columns to use

### Produces

A trained scikit-learn compatible model object (e.g., `LayeredCompModel`) with `fit` and `predict` methods.

## Prediction

### Accepts
A pandas dataframe

A weight_falloff hyperparameter.

### Produces

A "prediction" field decorated on the dataframe and returns it.

# Variable classification

We will need to be able to determine whether a column is numeric or categorical so the correct segmentation test can be applied.

# Segmentation Scoring

Uses a simple linear regression under the hood to fit sale price vs mean and evaluates the split quality by calculating the reduction in Mean Absolute Error (MAE).

1. Calculate the base MAE for the current set of data by fitting a linear regression (sale price vs mean) and calculating the MAE ($MAE_{total}$).

## Categorical

1. For each variant in the categorical (one-vs-rest), treating missing values as a distinct category:
   1. filter to only that variant, calculate its MAE vs its mean ($MAE_v$), and get its count ($N_v$).
   2. filter to the inverse of that variant, calculate its MAE vs its mean ($MAE_{inv}$), and get its count ($N_{inv}$).
   3. Calculate the weighted MAE for the split: $MAE_{weighted} = (MAE_v \times N_v + MAE_{inv} \times N_{inv}) / N_{total}$.
   4. The segmentation score is the ratio: $Score = MAE_{weighted} / MAE_{total}$.
2. Find the lowest segmentation score among all variants. In case of ties, choose the variant that splits the count most evenly. If still tied, choose the first one.

## Numeric 
1. Exclude NaNs from numeric features during this process.
2. Set num_iterations to minimum of 10 and log2(current population size).
3. Perform a binary search for an optimal breakpoint:
   1. Set the initial midpoint to the median of the feature values.
   2. filter to the entries below the midpoint, calculate its MAE ($MAE_{low}$) and count ($N_{low}$).
   3. filter to the entries above the midpoint, calculate its MAE ($MAE_{high}$) and count ($N_{high}$).
   4. Calculate the weighted MAE for the split: $MAE_{weighted} = (MAE_{low} \times N_{low} + MAE_{high} \times N_{high}) / N_{total}$.
   5. The segmentation score is the ratio: $Score = MAE_{weighted} / MAE_{total}$.
   6. Move the midpoint to the side that resulted in a better scoring split (lower ratio) and reform the "above" and "below" subsets for each step of the search.
4. Return the lowest segmentation score and corresponding midpoint found. In case of ties, choose the split that splits the count most evenly. If still tied, choose the first one.

# Node Size Constraints

Minimum node size is 2. Do not attempt to split a node if its size is below this threshold.
