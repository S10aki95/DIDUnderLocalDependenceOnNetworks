# Implementation Details

This document describes the detailed implementation of simulation experiments and real data analysis methods. For general project information and execution instructions, see the [main README](../README.md).

---

# 1. Real Data Experiments

## 1.1. China's SEZ Policy

### 1.1.1. Data Source

- **Source**: Xu (2025) "Difference-in-Differences with Interference"
- **Data Period**: Two-period panel data from 2004 (before policy implementation) and 2008 (after policy implementation)
- **Geographic Scope**: Village-level data in China (clustered at county level)
- **Policy**: Introduction of Special Economic Zone (SEZ) policy (SEZs established in 2005-2008, focusing particularly on provincial-level SEZs in 2006)
- **Sample Size**: Approximately 60,000 villages (SEZ-designated villages: ~4,000, non-SEZ villages: ~56,000)

### 1.1.2. Key Variables

- **Treatment Variable**: `W` - Introduction of SEZ policy (1: introduced, 0: not introduced)
- **Exposure Variable**: `G` - SEZ introduction status of other villages in the same county (0: low exposure, 1: high exposure)
  - Since detailed geographic information is not available in the data, county is defined as neighborhood
- **Outcome Variables**: Analysis is performed for the following 3 economic indicators (all in logarithms):
  - **Capital (log)**: `k` - Capital of enterprises in the village (2004: `k_2004`, 2008: `k_2008`)
  - **Employment (log)**: `l` - Employment of enterprises in the village (2004: `l_2004`, 2008: `l_2008`)
  - **Output (log)**: `y` - Output of enterprises in the village (2004: `y_2004`, 2008: `y_2008`)
- **Covariates**:
  - Basic covariates: `airport_p`, `port_p`, `kl_p`, `num_p`
  - County-level mean covariates: `airport_p_county_mean`, `port_p_county_mean`, `kl_p_county_mean`, `num_p_county_mean`

### 1.1.3. Exposure Mapping Definition

- **Leave-one-out SEZ ratio**: Calculate the SEZ introduction rate among other villages in the county to which village $i$ belongs, excluding village $i$ itself
  - Formula: $LOO\_SEZ\_Ratio_i = \frac{\text{Number of SEZ villages in county} - W_i}{\text{Total number of villages in county} - 1}$
- **Exposure Level**:
  1. For each village, calculate the SEZ introduction rate of other villages in the same county, excluding that village
  2. **Threshold**: Mean value of LOO ratios across all villages (calculated dynamically based on data)
  3. Classify as high exposure (G=1) if above threshold, low exposure (G=0) if below threshold
  4. If there is only one village in a county, set LOO ratio to 0

## 1.2. Implementation Methods

### 1.2.1. Comparison Methods

- Both methods from Xu (2025)
  1. **Doubly Robust Estimator (DR)**: Proposed method in Xu's paper
     - Utilizes both propensity score model and outcome model
     - Has consistency if either one is correctly specified
  2. **IPW Estimator (IPW)**: Comparison method in Xu's paper
     - Estimator that uses only propensity score
     - Requires propensity score model to be correctly specified

#### Estimation Model Details

**Propensity Score Model**:

- **Model**: Logistic regression
- **Covariates Used**: Following the paper's specification, the following 8 variables are used:
  - 4 basic covariates: `airport_p`, `port_p`, `kl_p`, `num_p`
  - 4 county-level Leave-one-out mean values: `airport_p_county_mean`, `port_p_county_mean`, `kl_p_county_mean`, `num_p_county_mean`
  - County-level mean values are calculated as Leave-one-out averages excluding the village's own value: $(Sum - Z_i) / (Count - 1)$
- **Propensity Scores Estimated**:
  - $\eta(Z_i) = P(W_i=1 | Z_i)$: Treatment probability
  - $\eta_{1g}(Z_i) = P(G_i=g | W_i=1, Z_i)$: Exposure level probability in treatment group
  - $\eta_{0g}(Z_i) = P(G_i=g | W_i=0, Z_i)$: Exposure level probability in control group
- **Clipping**: Propensity scores are clipped to [0.05, 0.95] range (default setting)

**Outcome Model**:

- **Model**: Linear Regression
- **Dependent Variable**: $\Delta Y_i = Y_{i2} - Y_{i1}$ (difference between time points)
- **Features**: Following the paper's specification, the following features are used:
  - $W_i$, $G_i$: Treatment variable and exposure variable
  - 4 basic covariates: `airport_p`, `port_p`, `kl_p`, `num_p`
  - Interaction terms: $W_i \times X_i$ (interactions with each basic covariate, 4 in total)
- **Estimation**: Estimate a model to predict $\Delta Y$ for each $(W, G)$ combination

### 1.2.2. Estimation Targets

**Xu (2025) Method Estimation Targets**:

- **DATT_0, DATT_1**: Direct effects at each exposure level (G=0, G=1) (Direct Average Treatment Effect on the Treated)
  - Direct effects at each exposure level in the treatment group (W=1)
- **ODE_DR, ODE_IPW**: Overall direct effects (weighted average) for each method
  - Formula: $ODE = \sum_{g \in \{0,1\}} DATT(g) \times P(G_i=g | W_i=1)$
- **Spillover Effects (based on definition in Appendix B of the paper)**:
  - **Spillover_Treated_DR, Spillover_Treated_IPW**: Spillover effects in treatment group (W=1)
    - Formula: $\tau(1,1,0) = E[Y_{i2}(1,1) - Y_{i2}(1,0) | W_i=1]$
    - Effects due to differences in exposure levels, fixing treatment status
  - **Spillover_Control_DR, Spillover_Control_IPW**: Spillover effects in control group (W=0)
    - Formula: $\tau(0,1,0) = E[Y_{i2}(0,1) - Y_{i2}(0,0) | W_i=0]$
    - Effects due to differences in exposure levels, fixing control status
- **Direct Effect Heterogeneity (maintained for backward compatibility)**:
  - **Direct_Effect_Heterogeneity_DR, Direct_Effect_Heterogeneity_IPW**: Direct effect heterogeneity in treatment group
    - Formula: $DATT(1) - DATT(0)$
    - This shows the difference in direct effects between different exposure levels within the treatment group
  - **Spillover_Effect_DR, Spillover_Effect_IPW**: Maintained for backward compatibility, same value as the direct effect heterogeneity above
- **Standard Errors**: Estimated using cluster bootstrap method (county-level clustering)
  - Default bootstrap iterations: 100 (configurable via `n_bootstrap` parameter in `settings/config.py`)
  - Each bootstrap sample performs resampling at county (county) level, including all villages within that county

**Proposed Methods (ADTT/AITT)**:

- **ADTT**: Average Direct Treatment Effect on the Treated
  - Formula: $\hat{\tau}_{ADTT} = \frac{1}{N} \sum_{i=1}^N \frac{D_i - \hat{e}_i}{\widehat{\pi}_i(1 - \hat{e}_i)} \Delta Y_i$
  - ADTT propensity score: $e_i = \mathrm{Pr}(D_i=1 \mid Z_i, \mathbf{D}_{N_A(i;K)})$
  - Marginal treatment probability: $\pi_i = \mathrm{Pr}(D_i=1 \mid Z_i)$
  - Real data uses county-level neighborhood definition (other villages in the same county are defined as neighbors)
  - Variable name conversion: In real data, W (treatment variable) is converted to D for calculation
- **AITT**: Average Indirect Treatment Effect on the Treated
  - Formula: $\hat{\tau}_{AITT} = \frac{1}{N} \sum_{i=1}^N \left[ \frac{1}{|N_A(i;K)|} \sum_{j \in N_A(i;K)} \frac{D_i - \hat{e}^\prime_{ij}}{\widehat{\pi}_i(1 - \hat{e}^\prime_{ij})} \Delta Y_j \right]$
  - AITT propensity score: $e^\prime_{ij} = \mathrm{Pr}(D_i=1 \mid Z_i, Z_j, D_j, \mathbf{D}_{N_A(j;K)}^{-i})$
  - Marginal treatment probability: $\pi_i = \mathrm{Pr}(D_i=1 \mid Z_i)$
  - Real data uses county-level neighborhood definition
  - For computational efficiency, when the number of neighbors exceeds `max_neighbors` (default: 10 for real data, 20 for simulation), neighbors are selected via random sampling
- **Standard Errors**: Since spatial coordinate information is not available in real data, HAC standard errors are not used; calculated from standard deviation of influence function
  - Formula: $SE = \frac{\mathrm{std}(Z_i)}{\sqrt{N}}$ (where $Z_i$ is the influence function)
  - For ADTT: $Z_i = \frac{D_i - e_i}{\widehat{\pi}_i(1 - e_i)} (Y_{2i} - Y_{1i})$
  - For AITT: $Z_i = \frac{1}{|N_A(i;K)|} \sum_{j \in N_A(i;K)} \frac{D_i - e'_{ij}}{\widehat{\pi}_i(1 - e'_{ij})} (Y_{2j} - Y_{1j})$

**Canonical DID (Standard DID, ignoring spillover effects)**:

- **Canonical_IPW**: Standard IPW-DID estimator (Abadie, 2005)
  - Ignores spillover effects (G), uses only W and covariates Z
  - Formula: $\hat{\tau}_{IPW} = \frac{1}{N} \sum_{i=1}^N \left(\frac{W_i - \hat{p}(Z_i)}{(1-\hat{p}(Z_i)) \cdot \frac{1}{N} \sum_{i=1}^N W_i} \right) \Delta Y_i$
  - Propensity score: $\hat{p}(Z_i) = P(W_i=1 | Z_i)$ (G is not used)
- **Canonical_DR_DID**: Standard DR-DID estimator
  - Ignores spillover effects (G), uses only W and covariates Z
  - Formula: $\hat{\tau}_{DR} = \hat{\eta}_{treat} - \hat{\eta}_{cont}$
  - Outcome model: $\Delta \hat{m}_0(Z_i) = E[\Delta Y_i | W_i=0, Z_i]$ (G is not used)
- **Importance**: Essential to demonstrate bias when spillover effects are ignored and to demonstrate the usefulness of DID that considers interference
- **Correspondence with Paper**: Also reported as a comparison target in Table 4 of the paper

**Note:** Since spatial coordinate information is not available in real data, HAC standard errors are not used. Standard errors for the proposed methods are calculated from the standard deviation of the influence function, and standard errors for Xu (2025) methods are estimated using cluster bootstrap method.

## 1.3. Execution Methods

### 1.3.1. Basic Execution

```bash
# Run real data experiment (default settings)
uv run python main.py --mode real_data

# Specify data directory and output directory
uv run python main.py --mode real_data --data_dir data --output_dir results
```

### 1.3.2. Output Files

- `results/sez_analysis_results.csv`: Analysis results (estimates and standard errors)
  - Contains results for each of the 3 outcome variables (capital, employment, output)
  - Records estimates and standard errors such as DATT_0, DATT_1, ODE, Spillover_Effect for each outcome variable
- `results/sez_analysis_report.md`: Detailed results report
  - Displays data summary and estimation results sectioned by each of the 3 outcome variables (capital, employment, output)
  - Explicitly states variable names used for each outcome variable
  - Organizes results in a format corresponding to Table 4 of Xu (2025) paper

---

# 2. Simulation Design and Implementation

## 2.1. Data Generating Process (DGP)

- Basic Settings
  - Total number of units: $N = 500$
  - Interference range distance: $K = 1.0$
  - Spatial size: $20.0 \times 20.0$
  - Number of simulation iterations: $M = 100$ (default value)
  - Random seed: Fixed at `42`
- Units
  - Generate $N$ units from $(x_i, y_i) \sim U(0, 20) \times U(0, 20)$
  - Adjust so that average degree of network neighborhood $N_{A(i;K)}$ is appropriate
- Confounding Variables:
  - Individual attributes: $z_i \sim N(0,1)$
  - Spatially correlated unobserved factors: $\mathbf{z}_{u} \sim N(\mathbf{0}, \{0.5^{l_A(i, j)} \})$
- Treatment Assignment (conservatively set to avoid errors)
  Treatment depends on both $z_i$ and $z_{u,i}$ and has spatial correlation.
    $$Pr(D_i=1 \mid z_i, z_{u,i}) = logit^{-1}(0.3 z_i + 0.8 z_{u,i})$$
- Outcomes
  - $Y_{1i} = \beta_1 z_i + \beta_2 z_{u, i} + \epsilon_{1i}$
  - $Y_{2i} = \delta + Y_{1i} + \tau D_i + f(S_i) + \gamma_1 z_{u,i} + \gamma_2 z_i + \epsilon_{2i}$
- Parameter Settings
  - $\delta = 1.0$ (constant term)
  - $\beta_1 = 1.2$ (coefficient of z_i)
  - $\beta_2 = 0.5$ (coefficient of z_u,i)
  - $\tau = 0.8$ (direct effect)
  - $\gamma_1 = 0.1$ (coefficient of z_u,i at time point 2)
  - $\gamma_2 = 0.2$ (coefficient of z_i at time point 2)
  - $\epsilon_{1i} \sim^{(i.i.d)} N(0, 1)$
  - $\epsilon_{2i} \sim^{(i.i.d)} N(0, 1)$
  - $f(\cdot)$: Function that outputs according to the spillover structure below

> Spillover Structure

| $S_i$ (Number of treated neighbors) | 0    | 1    | 2    | 3    |
| :---------------------------------- | :--- | :--- | :--- | :--- |
| Spillover Effect                    | 0.0  | 0.8  | 1.6  | 2.4  |
| (Level Index G)                     | 0    | 1    | 2    | 3    |

## 2.2. Comparison Scenarios

### 2.2.1. Exposure Mapping Specification Situations

We compare the proposed method with Xu (2025)'s DR/IPW estimators in the following scenarios, mimicking how researchers assume exposure mapping $G_i$.

| Scenario Name   | Abbreviation  | Description                                                                                 | Assumed Exposure Mapping $G_i$ Definition                        |
| :-------------- | :------------ | :------------------------------------------------------------------------------------------ | :--------------------------------------------------------------- |
| Proposed Method | Proposed      | Nonparametric estimation. Does not assume mapping.                                          | (Not assumed)                                                    |
| Xu (Case A)     | CS            | Correctly specified. Fully captures true 4-level structure.                                 | $G_{i}^{CS}$: Order aligned with spillover structure (0,1,2,3)   |
| Xu (Case B)     | MO            | Mis-specification of structure/order. 30% of units randomly reassigned to different groups. | **$G_{i}^{MO}$: Based on $G_{i}^{CS}$, 30% randomly reassigned** |
| Xu (Case C)     | FM            | Mis-specification of number of levels. Overly simplified (only strong exposure).            | $G_i^{FM} = \mathbb{I}(S_i > 1)$ (2 levels)                      |
| Standard Method | Canonical     | Standard DID. Completely ignores interference.                                              | (Ignored)                                                        |
| Modified TWFE   | Modified TWFE | Extended TWFE that considers interference existence with simple interaction term            | Uses $I(S_i≥1)$                                                  |

### 2.2.2. Evaluation Methods

> ① Point Estimation Accuracy Comparison (Bias & RMSE)

- Comparison Method:
  - ADTT vs. ODE: DATT(g) obtained from Xu's method is weighted averaged using treatment group's exposure level distribution to calculate Overall Direct Effect (ODE).
  - Compare proposed method's ADTT with ODE from each scenario (CS, MO, FM) and ATT from standard methods.
  - AITT: AITT is a parameter unique to this study, so direct comparison with other methods is not performed.

> ② Interval Estimation Validity Comparison (Coverage Rate)

- Comparison Method:
  - Comparison targets are limited to proposed method (ADTT) and standard methods (Canonical IPW, etc.).
  - Xu (2025)'s methods are not included in this coverage comparison.
  - Create a table summarizing Coverage Rate (95% confidence interval) in the following format.

## 2.3. Estimator Implementation

### 2.3.1. Proposed Methods (Proposed ADTT/AITT)

This is a nonparametric IPW estimator proposed in this study. Its characteristic is that it does not specify the functional form of exposure mapping and uses the "pattern" of neighboring treatment vectors themselves as conditioning.

> (1) ADTT Estimator ($\hat{\tau}_{ADTT}$)

ADTT is the average direct effect of treatment on treated units. Based on Theorem 2.1 and equation (3.2) of the paper, the estimator is constructed as follows.

$$\hat{\tau}_{ADTT} = \frac{1}{N} \sum_{i=1}^N \frac{D_i - \hat{e}_i}{\widehat{\pi}_i(1 - \hat{e}_i)} \Delta Y_i$$

Here, $\hat{e}_i$ and $\widehat{\pi}_i$ are the following estimators.

- ADTT propensity score: $e_i = \mathrm{Pr}(D_i=1 \mid Z_i, \mathbf{D}_{N_A(i;K)})$
- Marginal treatment probability: $\pi_i = \mathrm{Pr}(D_i=1 \mid Z_i)$

> (2) AITT Estimator ($\hat{\tau}_{AITT}$)

AITT is the average indirect effect that treatment on unit $i$ has on surrounding units $j$. Based on Theorem 2.2 and equation (3.2) of the paper, the estimator is constructed as follows.

$$\hat{\tau}_{AITT} = \frac{1}{N} \sum_{i=1}^N \left[ \frac{1}{|N_A(i;K)|} \sum_{j \in N_A(i;K)} \frac{D_i - \hat{e}^\prime_{ij}}{\widehat{\pi}_i(1 - \hat{e}^\prime_{ij})} \Delta Y_j \right]$$

Here, $\hat{e}^\prime_{ij}$ and $\widehat{\pi}_i$ are the following estimators.

- AITT propensity score: $e^\prime_{ij} = \mathrm{Pr}(D_i=1 \mid Z_i, Z_j, D_j, \mathbf{D}_{N_A(j;K)}^{-i})$
- Marginal treatment probability: $\pi_i = \mathrm{Pr}(D_i=1 \mid Z_i)$

> (3) Propensity Score Calculation Method

1. **Estimation of Marginal Treatment Probability π_i(z_i) = P(D_i=1|z_i)**
   - Estimated using logistic regression with only covariates $Z_i$
   - Data structure: Dataframe with unit $i$ as rows ($N$ rows)
   - Dependent variable: $y = D_i$
   - Explanatory variables: $X = [Z_i]$
   - Uses only observable covariates $Z_i$, does not condition on neighboring treatment vectors

2. **Feature Engineering**
   - When the dimension of explanatory variables differs by neighborhood size for each propensity score prediction, sort by distance from a specific unit,
   - If neighborhood size (number of units with $l_A < K$ from a specific unit) is `max_neighbors` (default: 10) or more: randomly sample `max_neighbors` units
   - If neighborhood size is less than `max_neighbors`: use all neighbors (with zero padding if needed)
   - `max_neighbors` is configurable in `settings/config.py` (default: 10)

3. **Model Training and Prediction (using logistic regression)**
   - ADTT (estimation of $e_i$):
     - Dependent variable: $y = D_i$
     - Explanatory variables: $X = [Z_i, \mathbf{D}_{N_A(i;K)}]$
   - AITT (estimation of $e^\prime_{ij}$):
     - Data structure: Create a dataframe (edge list format) with neighboring pairs $(i, j)$ of influencing unit $i$ and receiving unit $j$ as rows. Note that computational complexity becomes large.
     - Dependent variable: $y = D_i$
     - Explanatory variables: $X = [Z_i, Z_j, D_j, \mathbf{D}_{N_A(j;K)}^{-i}]$

4. **Implementation Notes**
   - π_i: Marginal treatment probability (conditioned only on $Z_i$)
   - e_i: ADTT propensity score (conditioned on $Z_i$ and neighboring treatment vector $\mathbf{D}_{N_A(i;K)}$)
   - e'_ij: AITT propensity score (conditioned on $Z_i$, $Z_j$, $D_j$, $\mathbf{D}_{N_A(j;K)}^{-i}$)

5. **Computational Efficiency in Real Data Analysis**
   - **Neighbor Sampling**: In real data, when the number of villages in a county is large (maximum 753 villages), the number of pairs becomes very large for AITT estimation (approximately 11.3 million pairs), so neighbor sampling is implemented for computational efficiency.
     - For each unit $i$, when the number of neighbors exceeds `max_neighbors`, select at most `max_neighbors` neighbors via random sampling.
     - Default setting: `max_neighbors=20` for simulation, `max_neighbors=10` for real data (configurable in `settings/config.py`)
     - Random seed: `random_seed=42` (configurable in `settings/config.py` for reproducibility)
     - This reduces total number of pairs from approximately 11.3 million to approximately 1.2 million, significantly shortening computation time.

> Standard Errors for Proposed Methods: HAC Estimator

- Variance estimator:$$\hat{V}_n = \sum_{s \geq 0} \omega\left(\frac{s}{b_n}\right) \hat{\Omega}_n(s)$$
  - $\omega(\cdot)$ is a kernel function (e.g., Bartlett kernel), $b_n$ is a bandwidth parameter.
  - $\hat{\Omega}_n(s)$ is the sample autocovariance between units separated by distance $s$.
    $$
    \hat{\Omega}_n(s) = \frac{1}{n} \sum_{i \in D_n} \sum_{j \in N_n^{\partial}(i ; s)}\left(Z_{i}-\bar{Z}_n\right)\left(Z_{j}-\bar{Z}_n\right)
    $$
- Definition of $Z_i$:
  - For ADTT: $Z_i = \frac{D_i - e_i}{\widehat{Pr}(D_i = 1)(1 - e_i)} (Y_{2i} - Y_{1i})$
  - For AITT: $Z_i = \frac{1}{|N_A(i;K)|} \sum_{j \in N_A(i;K)} \frac{D_i - e'_{ij}}{\widehat{Pr}(D_i = 1)(1 - e'_{ij})} (Y_{2j} - Y_{1j})$

- Kernel function: Bartlett kernel $\omega(x) = 1-|x|$ (where $|x| \le 1$), bandwidth $b_n$ is set to a fixed value (e.g., 2 times the interference range $K$)

### 2.3.2. Xu (2025) Estimators

We implement the doubly robust estimator and IPW estimator proposed in Xu (2025) as comparison targets.

> (1) Doubly Robust Estimator (DR)

The doubly robust estimator proposed in Xu (2025) is expressed by the following formula.

$$
\tau^{dr}(g) = \mathbb{E}_{D} \left[ \frac{W_{i}}{\eta(z_{i})} \frac{\mathbb{I}\{G_{i}=g\}}{\eta_{1g}(z_{i})} \left( (Y_{i2}-m_{2,1g}(z_{i})) - (Y_{i1}-m_{1,1g}(z_{i})) \right) \right. \\
\left. - \frac{1-W_{i}}{1-\eta(z_{i})} \frac{\mathbb{I}\{G_{i}=g\}}{\eta_{0g}(z_{i})} \left( (Y_{i2}-m_{2,0g}(z_{i})) - (Y_{i1}-m_{1,0g}(z_{i})) \right) \right. \\
\left. + \Delta m_{1g}(z_{i}) - \Delta m_{0g}(z_{i}) \right]
$$

**Implementation Details**:

- **Outcome Model**: Estimate separate linear regression models for treatment group ($D=1$) and control group ($D=0$) for each exposure level $g$
- **Prediction Calculation**: Regress $\Delta Y$ on $Z, D, G$ and calculate predicted values for each unit
- **Doubly Robust Estimator**: Estimator that utilizes both propensity score and outcome model
- **Advantage**: Has consistency if either model is correctly specified (doubly robust property)

> (2) IPW Estimator (IPW)

$$
\hat{\tau}^{ipw}(g) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{D_i - \hat{p}(Z_i)}{\hat{p}(Z_i)(1 - \hat{p}(Z_i))} \frac{\mathbb{1}\{G_i = g\}}{D_i\hat{\pi}_{1g}(Z_i) + (1 - D_i)\hat{\pi}_{0g}(Z_i)} (Y_{i2} - Y_{i1}) \right]
$$

**Implementation Steps:**

1. **Exposure Mapping Definition**:
   Define exposure mapping $G_i$ in 3 scenarios (CS, MO, FM) according to simulation settings.

2. **Propensity Score Model Estimation**:
   Estimate the following 3 types of propensity scores to calculate weights for IPW estimator.

3. **Estimator Calculation**:
   Calculate effect $\hat{\tau}^{ipw}(g)$ at each exposure level $g$ based on the above formula.

4. **Overall Direct Effect (ODE) Calculation**:
   Weight average each $\hat{\tau}^{ipw}(g)$ using exposure level distribution in treatment group to calculate Overall Direct Effect (ODE) as the final comparison metric.

   $$\widehat{ODE} = \sum_{g \in \mathcal{G}} \hat{\tau}^{ipw}(g) \times \hat{P}(G_i=g \mid D_i=1)$$

### 2.3.3. Standard Methods (Benchmark)

> (1) Canonical IPW

Standard IPW-DID estimator that considers covariates to estimate ATT.

$$\hat{\tau}_{IPW} = \frac{1}{N} \sum_{i=1}^N \left(\frac{D_i - \hat{p}(Z_i)}{(1-\hat{p}(Z_i)) (\frac{1}{N} \sum_{i=1}^N D_i)} \right) \Delta Y_i$$

- Propensity score: $\hat{p}(Z_i) = \mathrm{Pr}(D_i=1 \mid Z_i)$. Conditions only on covariates $Z_i$ to ignore interference.

> (2) Modified TWFE (Modified Two-Period Fixed Effects Model)

Extended TWFE model that considers interference, discussed in Xu (2025)

$$\Delta Y_i = \beta_0 + \tau_1 D_i + \tau_2 (1-D_i)S_i + \tau_3 D_i S_i + \beta_1^\prime Z_i + \epsilon_i$$

where $S_i = \mathbb{I}(\text{Number of treated neighbors} > 0)$

- **Direct effect at exposure level 0 ($S_i=0$)**: $\hat{\tau}_1$
- **Direct effect at exposure level 1 ($S_i=1$)**: $\hat{\tau}_1 + \hat{\tau}_3 - \hat{\tau}_2$

Final effect is calculated by weighted averaging these direct effects using exposure level distribution in treatment group ($D_i=1$) (proportion of $S_i=0$ and $S_i=1$).

> (3) Doubly Robust Difference-in-Differences Estimators

1. **Outcome Model**: Expected value of outcome change in control group ($D_i=0$) conditioned on covariates $X_i$: $\Delta m_0(X_i) = E[\Delta Y_i | D_i=0, X_i]$.
2. **Propensity Score Model**: Probability of belonging to treatment group ($D_i=1$) conditioned on covariates $X_i$: $p(X_i) = P(D_i=1 | X_i)$.

**Estimator Calculation**:
Calculate separately for treatment and control groups and take the difference:

$$\hat{\tau}_{DR} = \hat{\eta}_{treat} - \hat{\eta}_{cont}$$

where:

- $\hat{\eta}_{treat} = \frac{\sum_{i=1}^N D_i \hat{p}(X_i) (\Delta Y_i - \Delta \hat{m}_0(X_i))}{\sum_{i=1}^N D_i \hat{p}(X_i)}$
- $\hat{\eta}_{cont} = \frac{\sum_{i=1}^N (1-D_i) \frac{\hat{p}(X_i)}{1-\hat{p}(X_i)} (\Delta Y_i - \Delta \hat{m}_0(X_i))}{\sum_{i=1}^N (1-D_i) \frac{\hat{p}(X_i)}{1-\hat{p}(X_i)}}$

※ Implementation based on `drdid_panel` function from R package "DRDID" (Sant'Anna & Zhao, 2020)

---

# 3. Detailed Description of Result Files

This section provides detailed explanations of the purpose and content of each file generated in the `results/` directory.

## 3.1. Simulation Result Data Files (CSV)

| File Name                                       | Purpose                                                                           | Main Content                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `simulation_results.csv`                        | Records estimates and standard errors of all estimators for each simulation trial | For each trial (row), includes estimates and standard errors of proposed methods (ADTT, AITT), standard methods (Canonical IPW, Canonical TWFE, DR-DID, Modified TWFE), and Xu (2025) methods (DR/IPW estimators for CS, MO, FM scenarios). Also includes true parameter values (`true_adtt`, `true_aitt`) and number of units (`n_units`). Used as raw data to evaluate distribution and bias of each estimator. |
| `simulation_results_neighbor_features_15.csv`   | Records simulation results when neighbor feature count is set to 15               | Same structure as `simulation_results.csv` but contains only results with `max_neighbors=15` setting. Used as part of sensitivity analysis to evaluate impact of hyperparameter (neighbor feature count) selection on estimator performance.                                                                                                                                                                      |
| `evaluation_results.csv`                        | Summarizes quantitative evaluation results of each estimator's performance        | For each estimator, includes results of the following evaluation metrics: **Bias** (difference between estimator's mean value and true value), **RMSE** (root mean squared error), **N_Valid** (number of valid estimates), **Coverage_Rate** (95% confidence interval coverage rate, only for proposed and standard methods). Provides main metrics for comparing estimator performance.                         |
| `evaluation_results_neighbor_features_15.csv`   | Records evaluation results when neighbor feature count is set to 15               | Same structure as `evaluation_results.csv` but contains only evaluation results with `max_neighbors=15` setting. Used as sensitivity analysis results.                                                                                                                                                                                                                                                            |
| `robustness_results.csv`                        | Records integrated results of robustness experiments                              | For each experiment type (`experiment_type`: sample, network, spatial, neighbor) and experiment ID (`experiment_id`), records Bias, RMSE, Coverage_Rate for each estimator. Also includes experiment settings (`config_name`, `overrides`), making it possible to track which settings were used. Used as comprehensive dataset to evaluate robustness of estimators.                                             |
| `coverage_rate.csv`                             | Summarizes 95% confidence interval coverage rates for each estimator concisely    | Consists of 2 columns: estimator name (`Estimator`) and coverage rate (`Coverage Rate (95% CI)`). Includes coverage rates for proposed methods (ADTT, AITT) and standard methods (Canonical IPW, Canonical TWFE, DR-DID, Modified TWFE). Xu (2025) methods are not subject to interval estimation, so displayed as N/A. Used to quickly verify validity of interval estimation for estimators.                    |
| `se_comparison_table{experiment_id_suffix}.csv` | Compares bootstrap standard errors with regular standard errors                   | Generated when `--use_bootstrap` flag is enabled. Table comparing regular standard errors (influence function-based or HAC) with bootstrap standard errors for each estimator. Used to check differences due to standard error estimation methods.                                                                                                                                                                |

## 3.2. Real Data Analysis Result Files

| File Name                   | Purpose                                                                                                         | Main Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sez_analysis_results.csv`  | Records estimation results for each outcome variable (capital, employment, output) in China SEZ policy analysis | For each outcome variable (`outcome_var`, `outcome_name`), includes the following estimates and standard errors: **DATT_DR_0, DATT_DR_1** (direct effects at each exposure level from Xu (2025)'s DR estimator), **DATT_IPW_0, DATT_IPW_1** (direct effects at each exposure level from IPW estimator), **ODE_DR, ODE_IPW** (overall direct effects), **Spillover_Treated_DR, Spillover_Control_DR** (spillover effects in treatment/control groups, DR estimator), **Spillover_Treated_IPW, Spillover_Control_IPW** (spillover effects in treatment/control groups, IPW estimator), **Canonical_IPW, Canonical_DR_DID** (standard DID estimators). Provides main results of real data analysis in table format. |
| `sez_analysis_report.md`    | Reports detailed results of real data analysis in readable format                                               | Displays experiment purpose and setting descriptions, data summary (sample size, variable descriptions, etc.), and analysis results sectioned by each outcome variable (capital, employment, output). Organizes results in format corresponding to Table 4 of Xu (2025) paper. Detailed tables including estimates, standard errors, and statistical significance. Used as reference material when using real data analysis results in papers or presentations.                                                                                                                                                                                                                                                  |
| `missing_data_analysis.csv` | Records missing data situation in real data analysis                                                            | For each outcome variable (`outcome_var`, `outcome_name`), includes the following information: `n_villages_2004` (number of villages in 2004), `n_villages_2008` (number of villages in 2008), `n_villages_both` (number of villages with data at both time points), `n_after_merge` (number of villages after merge), `n_complete` (number of villages with complete data), `n_after_dropna` (number of villages after removing missing values), `xu_paper_n` (sample size reported in Xu (2025) paper), `difference` (difference in sample size from paper). Used for data quality management and comparison with paper.                                                                                       |

## 3.3. Visualization Files (PNG)

### Simulation-Related Figures

| File Name                         | Purpose                                                                            | Main Content                                                                                                                                                                                                                                                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spillover_structure.png`         | Visualizes spillover structure generated in simulation                             | Displays distribution of number of treated neighbors ($S_i$) for each unit as histogram. Can visually confirm stepwise changes in spillover effects (0, 1, 2, 3+ treated neighbors). Used to understand characteristics of data generating process.                                                                           |
| `estimator_distribution_adtt.png` | Visually compares distribution of each estimator                                   | Displays distribution of estimates for each estimator (proposed methods, standard methods, Xu (2025) methods) as box plots. True parameter value (ADTT = 0.8) is displayed as red horizontal line, allowing visual confirmation of bias and variance of each estimator. Used to intuitively understand estimator performance. |  |
| `units_scatter_plot.png`          | Visualizes spatial placement and treatment status of units generated in simulation | Displays units placed in 2D space (20.0 × 20.0) as scatter plot. Treatment group (W=1) colored red, control group (W=0) colored blue. Can confirm spatial clustering patterns and treatment distribution. Used to understand spatial characteristics of data generating process.                                              |

### Robustness Analysis Figures

| File Name                            | Purpose                                                                                                              | Main Content                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `robustness_bias_vs_sample_size.png` | Visualizes changes in bias of each estimator with respect to sample size changes                                     | Plots sample size (N=300, 500, 700) on x-axis, bias on y-axis. Displays bias of each estimator as line graph, confirming tendency for bias to decrease as sample size increases. Used to evaluate asymptotic properties (consistency) of estimators.                                                                                 |
| `robustness_rmse_vs_sample_size.png` | Visualizes changes in RMSE of each estimator with respect to sample size changes                                     | Plots sample size on x-axis, RMSE on y-axis. Displays RMSE of each estimator as line graph, confirming tendency for RMSE to decrease as sample size increases. Used to evaluate efficiency of estimators.                                                                                                                            |
| `robustness_bias_vs_density.png`     | Visualizes changes in bias of each estimator with respect to network density (interference range distance K) changes | Plots network density (K=0.8, 1.0, 1.2) on x-axis, bias on y-axis. Displays bias of each estimator as line graph, confirming how estimator performance changes as network density increases (interference range widens). Used to evaluate impact of interference range.                                                              |
| `robustness_rmse_vs_density.png`     | Visualizes changes in RMSE of each estimator with respect to network density changes                                 | Plots network density on x-axis, RMSE on y-axis. Displays RMSE of each estimator as line graph, confirming how RMSE changes as network density increases. Used to evaluate impact of interference range.                                                                                                                             |
| `robustness_bias_vs_correlation.png` | Visualizes changes in bias of each estimator with respect to spatial correlation (confounding strength) changes      | Plots spatial correlation parameter (z_u_correlation_base=0.2, 0.5, 0.8) on x-axis, bias on y-axis. Displays bias of each estimator as line graph, confirming how estimator performance changes as spatial correlation strengthens (confounding strengthens). Used to evaluate robustness of estimators (robustness to confounding). |  |
| `robustness_rmse_vs_correlation.png` | Visualizes changes in RMSE of each estimator with respect to spatial correlation changes                             | Plots spatial correlation parameter on x-axis, RMSE on y-axis. Displays RMSE of each estimator as line graph, confirming how RMSE changes as spatial correlation strengthens. Used to evaluate robustness of estimators.                                                                                                             |

### Sensitivity Analysis Charts

| File Name                                             | Purpose                                                                                       | Main Content                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sensitivity_bias_vs_features.png`                    | Visualizes changes in bias of proposed methods with respect to neighbor feature count changes | Plots neighbor feature count (max_neighbors=5, 10, 15) on x-axis, bias on y-axis. Displays bias of proposed methods (ADTT, AITT) as line graph, confirming impact of hyperparameter selection on estimator performance. Used to evaluate robustness of hyperparameter selection. |
| `sensitivity_rmse_vs_features.png`                    | Visualizes changes in RMSE of proposed methods with respect to neighbor feature count changes | Plots neighbor feature count on x-axis, RMSE on y-axis. Displays RMSE of proposed methods (ADTT, AITT) as line graph, confirming impact of hyperparameter selection on estimator efficiency. Used to evaluate robustness of hyperparameter selection.                            |
| `bootstrap_se_distribution{experiment_id_suffix}.png` | Visualizes distribution of bootstrap standard errors                                          | Generated when `--use_bootstrap` flag is enabled. Displays distribution of bootstrap standard errors for each estimator as box plots. By comparing with regular standard errors, can visually confirm differences due to standard error estimation methods.                      |

## 3.4. Report Files (Markdown)

| File Name               | Purpose                                                                | Main Content                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `robustness_summary.md` | Provides comprehensive summary report of robustness experiment results | Displays experiment purpose and setting descriptions, results of each robustness experiment (sample size, network density, spatial correlation, neighbor feature count) sectioned. Organizes estimator performance (Bias, RMSE, Coverage Rate) for each experiment in table format. Includes main findings and discussion on estimator performance. Used as reference material when using robustness experiment results in papers or presentations. |

## 3.5. High-Quality Figure Generation with R

High-quality figures can be generated using R's ggplot2 with data generated after simulation execution (`plot_data_*.csv`).

### Prerequisites

1. R is installed (`Rscript` command is included in PATH)
2. Required R packages are installed:

   ```r
   install.packages(c("ggplot2", "dplyr", "tidyr", "argparse", "readr"))
   ```

### Usage

#### Direct R Script Execution (Recommended)

```bash
# Generate all figures (uses results/ directory by default)
Rscript src/visualize_r.R --type all

# Generate spillover structure figure
Rscript src/visualize_r.R --type spillover_structure

# Generate ADTT distribution figure (proposed methods vs benchmark)
Rscript src/visualize_r.R --type estimator_distribution_adtt --plot_type proposed_vs_benchmark --true_adtt 0.5

# Generate ADTT distribution figure (proposed methods vs Xu estimators)
Rscript src/visualize_r.R --type estimator_distribution_adtt --plot_type proposed_vs_xu --true_adtt 0.5

# Generate AITT distribution figure (true parameter values automatically loaded from CSV)
Rscript src/visualize_r.R --type estimator_distribution_aitt

# Generate unit placement scatter plot
Rscript src/visualize_r.R --type units_scatter

# Specify custom directories
Rscript src/visualize_r.R --type all --input_dir results --output_dir results/figures

# Specify figure size and resolution
Rscript src/visualize_r.R --type spillover_structure --width 14 --height 10 --dpi 300

# Display help
Rscript src/visualize_r.R --help
```

### Available Figure Types

| Type                          | Description                                      | Required Options                                        |
| ----------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| `all`                         | Generate all figures                             | None                                                    |
| `spillover_structure`         | Spillover effect count distribution              | None                                                    |
| `estimator_distribution_adtt` | ADTT estimator distribution                      | `--true_adtt` (automatically loaded if included in CSV) |
| `estimator_distribution_aitt` | AITT estimator distribution                      | `--true_aitt` (automatically loaded if included in CSV) |
| `units_scatter`               | Unit placement and treatment status scatter plot | None                                                    |

**Note**: If `true_adtt` or `true_aitt` are included in CSV files (`plot_data_*.csv`), mean values are automatically used. Use `--true_adtt` or `--true_aitt` options to specify explicitly.

### Output Files

Figures generated by R script are saved with `*_r.png` naming (e.g., `spillover_structure_r.png`). This distinguishes them from figures generated by Python (`*.png`).

### Troubleshooting

- **Rscript not found**: Verify that R is installed and included in PATH. Check with `Rscript --version`.
- **Package errors**: Verify that required R packages are installed.
- **Input files not found**: First run simulation to generate data (`uv run python main.py --mode simulation`).

---

# 4. References

- Xu, R. (2025). Difference-in-Differences with Interference. *arXiv preprint arXiv:2306.12003*.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *The Review of Economic Studies*, 72(1), 1-19.
- Kojevnikov, D., Marmer, V., & Song, K. (2021). Limit theorems for network dependent random variables. *Journal of Econometrics*, 222(2), 882-908.
