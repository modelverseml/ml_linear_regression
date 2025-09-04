# Linear Regression : The Foundation of Machine Learning

Linear regression is often considered the starting point for anyone beginning their machine learning journey. At first glance, it seems simple — and in many ways, it is. However, there are several constraints and assumptions behind the model that people often overlook when building it. Understanding these boundaries is essential to ensure the model’s reliability.

At its core, linear regression is about fitting a straight line to data. If the dataset contains only a single feature, the technique is called Simple Linear Regression. When multiple features are involved, it becomes Multiple Linear Regression.


### Assumptions of Linear Regression

For linear regression to give reliable results, certain assumptions must hold true:

- Linearity
  - There should be a linear relationship between the independent variable(s) and the dependent variable.
- Independence of Errors
  - The residuals (errors) should be independent of each other. In other words, one error should not influence another.
- Normality of Errors
  - The residuals should be normally distributed. This helps in making valid inferences about the coefficients.
- Homoscedasticity
  - The residuals should have constant variance across all levels of the independent variables. If variance changes (heteroscedasticity), predictions may become unreliable.
- No Multicollinearity
  - Independent variables should not be highly correlated with each other. Strong correlation among predictors makes it difficult to estimate the effect of each variable accurately.
- No Autocorrelation
  - Errors should not be correlated with themselves over time. Autocorrelation is common in time-series data and violates regression assumptions.
- Feature Relevance (practical consideration)
  - The chosen features should have meaningful predictive power. Irrelevant or noisy features can reduce model performance.

## Linearity 

### Checking the Relationship Between Features and Target

- **Simple Linear Regression (one feature):**  
  - Use a **scatter plot of X vs. Y**.  
  - If the pattern looks roughly linear, overlay a **best-fit line**.
- **Multiple Linear Regression (many features):**  
  A single scatter plot is not enough. Instead, use:
  -  **Pair plots / scatter-matrix** → quick glance at pairwise relations  
  -  **Correlation heatmap** → strength & direction of linear association

###  Fitting the Best Line

For two features x<sub>1</sub>, x<sub>2</sub> and output y , the model will be

ŷ = β<sub>0</sub> +  β<sub>1</sub> . x<sub>2</sub> +  β<sub>2</sub>.x<sub>2</sub> 

We chose  β<sub>0</sub>,β<sub>1</sub>, β<sub>2</sub such that to reduce sum of squared error.

SSE(β) = ∑<sub>i=1</sub><sup>n</sup> ( ŷ<sub>i</sub> - y<sub>i</sub> )<sup>2</sup> = ∑<sub>i=1</sub><sup>n</sup> ( β<sub>0</sub> +  β<sub>1</sub> . x<sub>1i</sub> +  β<sub>2</sub>.x<sub>2i</sub>  - y<sub>i</sub> )<sup>2</sup>

We solve for B using vectors

X = [1  x<sub>1</sub>  x<sub>2</sub> \]  Y = y

ŷ = βX and  β̂ ​= arg min <sub>β</sub> ​ || Xβ − y || <sup>2</sup>

- closed solution
  - β̂ ​=(X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>y​
- Iterative process : Gradient descent
  - βⱼ := βⱼ - α * ∂/∂βⱼ [ SSE(β) \]
  - The derivative of SSE with respect to βⱼ is: ∂/∂βⱼ [ SSE(β) \] = -2 ∑ ( yᵢ - ŷᵢ ) Xᵢⱼ
  - Substituting this into the update rule gives: βⱼ = βⱼ + 2α ∑ ( yᵢ - ŷᵢ ) Xᵢⱼ
  - We apply this update to all β coefficients in each iteration. By adjusting the learning rate α and repeating the process iteratively, we gradually minimize the error until the model converges.

**The code for these two methods was provided above as separate functions, please check it out.**

Those implementations were just to demonstrate how the actual calculations work. In practice, you don’t need to perform these steps manually—libraries like statsmodels and scikit-learn handle them automatically once you provide the input and output data.

## Error Independence:

What does error independece mean actuall ? 

Error independence means that the error (or residual) for one observation should not be correlated with the error of another observation.

For example, suppose we have two instances in the dataset:

- Instance 1 → (x1ᵢ, x2ᵢ, yᵢ)

- Instance 2 → (x1ⱼ, x2ⱼ, yⱼ)

The error for the first instance is:
eᵢ = ŷᵢ - yᵢ

And the error for the second instance is:
eⱼ = ŷⱼ - yⱼ

For independence, eᵢ should not provide any information about eⱼ. In other words, the error of one data point should not influence or predict the error of another.

To check this assumption, we usually:

- Plot the residuals against the order of observations (or time, if it’s time-series data).
- If the errors are independent → the plot should look like random scatter around zero with no visible pattern.
- If not → you may see trends, cycles, or clustering, which indicates correlation among errors.

## Normality of Error Terms:

It indicates that the errors (residuals) from predictions are random and follow a normal distribution.

This assumption matters because:

- It ensures that the statistical inference we do (like hypothesis testing and confidence intervals) is valid.

- If errors are normally distributed, the estimated coefficients (β) are unbiased and efficient.

We usually check this by:

- Histogram of residuals → should look bell-shaped.

## Homoscedasticity:

It means that the errors (residuals) should have constant variance across all levels of the predicted values (ŷ) or input features.

- If homoscedasticity holds → the spread of residuals remains roughly the same for all fitted values.

- If violated (called heteroscedasticity) → the variance of errors changes, often seen as a “funnel” or “cone” shape in residual plots.

Why it matters:

- Homoscedasticity ensures that the model’s estimates of standard errors are correct.

- If violated, hypothesis tests (t-tests, F-tests) and confidence intervals may become unreliable.

How to check:

- Residuals vs fitted values plot → should show random scatter with no clear pattern and roughly equal spread
