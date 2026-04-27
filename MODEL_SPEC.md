# Layered Comp Model

## Overview:

The idea is to build "hierarchical" predictions that start with a general predicted price, then refine the prediction to be more specific by adding information and narrowing the comparison group.

You take the "Wilson mean" of all your parcels, then you find a filter that splits them into the best submarkets you can find and produce a "child node" for each variant of that filter (using a one-vs-rest approach for categorical data). Then you repeat until you've filtered down to a single data point. The value is a weighted average that prioritizes comparing well against closer matches and comparing slightly less well against further matches.

To get the predictions back out, you find the most specific bucket your subject matches, then you trace its path up the tree, taking and weighting the Wilson means as you go.

## Method:

### Training

1. Build a tree.
2. Plot sale prices.
3. Take the Wilson mean: This is defined as the mean after trimming the top 2.5% and the bottom 2.5% (the middle 95%).
4. Find a set of filters to use (segmentation score based on weighted MAE reduction (MAE of sale prices vs the mean of the subset)).
5. Make child nodes (one-vs-rest for categorical, or binary split for numeric using binary search for the breakpoint). We choose the split that results in the lowest ratio of weighted child MAE to parent MAE.
6. Repeat from step 3 until we've filtered down to a single parcel (leaf node) or cannot split further (minimum node size = 2).

### Predicting

1. Find the node furthest down in the hierarchy that matches your parcel.
2. Note its Wilson mean and the Wilson means of all nodes above it in the hierarchy.
3. Calculate weights for each node in the path:
   - Use the formula $w(x)=(1−x)^{weight\_falloff}$.
   - $x$ is normalized from 0 to 1, evenly spaced by the depth of the node.
   - $x = 0$ for the most specific (leaf) node.
   - $x = 1$ for the root node.
4. Take the weighted average of the Wilson means.
5. There's your prediction.

### Hyperparameters

weight_falloff: 0 to 1. Will be used in w(x)=(1−x)^weight_falloff where x is normalized from 0 to 1


## Nuances:

The Wilson Mean keeps the prediction from going too crazy on the large sets, and it also penalizes the small sets so when the test set gets specific, the value won't swing wildly.

No parcel will get mapped to its own sale price because the weight falloff of the means will add some noise to it, in the direction of the broader market.

Every parcel should compare well because this model is fundamentally doing a hierarchical version of comp analysis to determine the value.

If a predicted parcel has a feature that wasn't in the training set, that particular level of nuance will be missed, but the parcel will still slot into a node slightly higher up the tree, so the model should still perform reasonably well even for things we don't have representative sales for.

The function that the weighted medians follows will determine a lot of how this model handles accuracy vs equity. A fast falloff will give good accuracy but may miss broader market trends. A slow falloff will promote "normativity" in predictions, but may miss market nuance and not assign correct values to particularly rare but valuable features.


# Layered Comp Bagging Model

## Overview:

A bagging ensemble version of the primary algorithm that reduces variance and automatically optimizes the `weight_falloff` for each tree in the ensemble.

## Method:

### Training

For each tree from 1 to `tree_count`:

1. **Subsampling**: Randomly sample a subset of the training data equal to `sample_pct` of the total records.
2. **Internal Split**: Divide the sample into a **training portion** and a **test portion**. The `sample_pct` also serves as the split ratio (e.g., if `sample_pct` is 0.8, 80% of the sample is for training, 20% for testing). If the test portion calculation results in 0, a minimum of 1 row is used.
3. **Tree Construction**: Build a standard `LayeredCompModel` tree using only the training portion.
4. **Weight Falloff Optimization**: Find the optimal `weight_falloff` (between 0 and 20) that minimizes the error (MAE or MSE) on the test portion. Since the error function typically has a single local minimum, use **Brent's method** or a binary search for optimization.
5. **Storage**: Save the tree structure and its specific optimized `weight_falloff`.

### Predicting

1. Generate predictions for the input from all `tree_count` individual trees.
2. Each tree uses its own optimized `weight_falloff` discovered during training.
3. The final prediction is the **arithmetic mean** of all individual tree predictions.

## Hyperparameters

*   **tree_count**: Integer (min 1, default 10). Number of trees to build.
*   **sample_pct**: Float (0 < x < 1, default 0.8). Fraction of data sampled for each tree and used as the internal split ratio.
*   **random_state**: Integer or RandomState instance for reproducibility.
*   **split_metric**: {'mae', 'mse'}. Metric used for both tree splitting and `weight_falloff` optimization.

