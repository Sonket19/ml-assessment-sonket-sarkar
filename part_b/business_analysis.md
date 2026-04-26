# Part B: Business Case Analysis

The scenario: a fashion retailer with 50 stores (urban, semi-urban, rural) runs one of five promotions per store per month - Flat Discount, BOGO, Free Gift, Category-Specific Offer, or Loyalty Points Bonus. They want to figure out which promotion to run in each store each month to maximise items sold.

---

## B1. Problem Formulation - 8 marks

### (a) What kind of ML problem is this? - 3 marks

This is a **supervised regression problem**.

- **Target variable:** `items_sold` - the number of units sold by a given store in a given month under a given promotion.
- **Candidate input features:** store-level stuff (`store_id`, `store_size`, `location_type`), promotion attributes (`promotion_type`, plus discount depth or duration if we have them), calendar features (`month`, `is_weekend`, `is_festival`, `is_month_end`), and contextual features (`competition_density`, local demographics, weather, footfall if available).
- **Why regression and not classification:** the target is a continuous count of units. Knowing the predicted *magnitude* of sales is what lets us actually rank promotions against each other. "High vs low" buckets would be too coarse to make a real decision with.
- **Why supervised:** we have years of historical data showing what promotion ran in which store-month and what items_sold actually came out. That's labelled training data.

The deployment-time question - "which of the five promotions should I run in store X next month?" - turns into: score all five candidate promotions for the upcoming month, predict items_sold for each, and pick the one with the highest predicted value.

### (b) Why items_sold is a more reliable target than total sales revenue - 3 marks

Total revenue mixes up two very different things: how much customers bought, and how much we discounted to make them buy it. A 50% Off promotion will almost always lose to a Free Gift on revenue per transaction, even if it actually moves more units, because the per-item price is just lower under a discount. If the model learns from revenue, it'll learn that deep discounts are bad - even when they're exactly the right tool for clearing slow-moving stock or driving footfall.

`items_sold` separates customer behavioural response from pricing mechanics. It answers a cleaner question: under this promotion, how many people decided to put a unit in their basket? That's the lift the marketing team is actually trying to drive, and it's also what the inventory and supply chain teams care about.

**The broader principle:** pick a target that measures the behaviour you want to drive, not a downstream financial outcome that's contaminated by the lever you're pulling. In real ML projects the target should be (i) directly tied to the action you're optimising, (ii) free of mechanical coupling with the input features, and (iii) measured consistently across all observations. Picking a financially convenient metric instead - revenue, margin, profit - usually introduces confounders that the model will exploit in the wrong direction.

### (c) Why a single global model across all 50 stores is the wrong default - 2 marks

The junior analyst's idea of one global model assumes the relationship between (`promotion_type`, calendar, competition) and `items_sold` is the same everywhere. In retail this is basically never true. A Free Gift promotion can do really well in a busy urban mall store with lots of impulse buyers, and totally flop in a rural store where shoppers come in deliberately and care about price. Forcing one model to fit both averages out the differences and leaves money on the table in both places.

**My alternative:** a hierarchical / mixed approach. Build one model that includes store identity in the features (one-hot `store_id`, plus `store_size`, `location_type`, demographic descriptors), but use a model class that can capture interactions - gradient boosting and random forests do this naturally; for a linear model you'd need explicit interaction terms like `promotion_type` x `location_type`. That gives you one model to maintain while still letting it learn store-specific responses. For really large stores or stores with very weird histories, you could layer a per-store model on top of the global one.

I'd reject 50 separate per-store models because most stores won't have enough history to fit reliably, and maintaining 50 models is a nightmare.

---

## B2. Data and EDA Strategy - 10 marks

### (a) How I'd join the tables and what the final grain looks like - 4 marks

There are four raw sources to combine.

**The joins:**
- `transactions` to `stores` on `store_id`, to attach `store_size`, `location_type`, demographic info. Inner join (every transaction must come from a known store).
- `transactions` to `promotions` on `(store_id, date)` or `(store_id, promotion_id)`, to attach `promotion_type` and any promotion attributes (depth, duration). Left join from the transactions side, because some transactions happened outside any active promotion - those rows get a `promotion_type = "none"` flag, which is itself useful information.
- `transactions` to `calendar` on `date`, to attach `is_weekend`, `is_festival`, `month`, `day_of_week` and any other date flags.

**Final grain:** one row = one store, one month, one promotion (assuming at most one promotion runs per store per month, which the scenario implies). Each row has identifiers (`store_id`, `month`, `year`), the decision being scored (`promotion_type`), store attributes, calendar/contextual features for that month, and the target `items_sold` summed across all transactions for that store-month.

**Aggregations I'd do before modelling:**
- Sum `items_sold` from transactions to month level per store.
- Count festival days within the month -> `num_festival_days`.
- Count weekend days within the month -> `num_weekend_days`.
- Take the month-end (or monthly average) `competition_density` if it varies daily.
- Lag features: same store's `items_sold` last month, and same store's `items_sold` in the same month a year ago. These are usually some of the strongest features in retail demand modelling because they capture the store's baseline trajectory.

### (b) EDA passes I'd run before modelling - 4 marks

Five things I'd actually look at, each tied to a specific decision.

1. **Histogram of `items_sold`, plus the log version.** Looking for skew and heavy tails. *Why it matters:* if the target is heavily right-skewed I'd model `log(items_sold)` instead of the raw value. Skewed targets bias linear models and let RMSE get dominated by a handful of huge outlier months.

2. **Boxplot of `items_sold` by `promotion_type`, faceted by `location_type`.** Looking for whether the ranking of promotions changes between location types. *Why it matters:* if the order of promotions flips between facets (e.g. BOGO best in urban, worst in rural), then I need explicit interaction features for a linear model. Tree-based models can pick this up on their own, but knowing about it ahead of time changes which model class I'd choose.

3. **Time-series plots of monthly `items_sold` per store for the biggest 12 stores.** Looking for trends, seasonality, and breaks (renovations, lockdown periods, new competitor opening nearby). *Why it matters:* tells me whether to add an explicit trend feature, year fixed effects, or whether some periods need to be dropped from training as anomalies.

4. **Correlation/mutual information matrix on numeric features vs `items_sold`.** *Why it matters:* shows me which features carry the most univariate signal, and which redundant pairs (e.g. `store_size` and `historical_avg_sales`) I should drop one of to avoid multicollinearity hurting linear models.

5. **Promotion frequency by store and by month.** Just counting how often each store has run each promotion. *Why it matters:* tells me whether there's enough data per `(store, promotion)` cell to learn store-specific effects. If Loyalty Bonus has only ever run twice in Store 7, those predictions will be low-confidence and probably need a fallback rule.

### (c) The 80% no-promotion imbalance - 2 marks

If 80% of transactions happen outside any promotion, the model will see "no promotion" as the dominant context and end up learning baseline demand really well, while only weakly learning the *uplift* each promotion gives. That's the opposite of what the business needs.

**What I'd do about it, in priority order:**

1. **Reframe the target as uplift instead of absolute sales.** For each store-month with a promotion, work out the predicted no-promotion baseline (from a baseline model, or from that same store's no-promotion months) and model `items_sold - baseline` instead. Now the model is learning directly from the 20% promoted observations, with the 80% no-promotion data fitting the baseline. This is the standard treatment-effect / uplift modelling framing.
2. **Stratified sampling in cross-validation** so each fold has a representative mix of promoted and non-promoted observations.
3. **Class-weighted loss** if I'm sticking with the direct prediction framing - up-weight the loss contribution from promoted observations so the model has to fit them as carefully as the baseline observations.
4. **Run two separate models** - a baseline model on the 80% no-promotion data, and a promotion-only model on the 20% promoted rows - and combine their predictions at scoring time. Less elegant than full uplift modelling but easier to implement and often performs comparably on moderate-sized data.

---

## B3. Model Evaluation and Deployment - 12 marks

### (a) Train/test setup with three years x 50 stores; why random split is wrong - 4 marks

**Setup:** the data is a panel - 50 stores observed monthly over 36 months, so roughly 1,800 store-month rows. The split needs to respect the time ordering and the panel structure.

**What I'd use: rolling-origin (time-series) cross-validation.**
- Train on months 1-24, validate on 25-27, test on 28-30. Then slide forward: train on 1-27, validate on 28-30, test on 31-33. And so on.
- All 50 stores appear in both train and test in every fold - I'm not holding out stores, I'm holding out *future months*. That matches the deployment task: predict next month for stores we already know about.
- The reported metric is the average across all the test periods.

**Why a random split is wrong:** random splits scatter rows from the three-year window into train and test indiscriminately. The training set ends up with rows from *after* some test rows, so the model is implicitly using future information to predict the past (e.g. a competitor opening in March 2024 shows up in March 2024 training rows but is being asked to predict February 2024). Test metrics come out optimistic, and then the model fails when actually deployed because deployment is purely forward-looking.

**Metrics:**
- **RMSE on items_sold per store-month** as the primary metric - it punishes big errors more, which is what matters when an inventory decision rests on the prediction.
- **MAE per store-month** as a secondary metric - easier to explain to non-technical people ("we're off by 38 units per store per month on average").
- **MAPE broken down by store size** - to check the model isn't systematically worse on small stores, where 10 units of error matters way more than in a big store.
- **Top-1 promotion accuracy** - the actual business question is *ranking* promotions, so for each held-out store-month, check whether the promotion the model would have recommended is actually the one that produced the highest realised sales. This is the metric that maps most directly to business value, even though it's harder to optimise.

### (b) Investigating the surprising recommendations using feature importance - 4 marks

Setup: the model recommends Loyalty Points Bonus for Store 12 in December and Flat Discount for Store 12 in March. The marketing team is sceptical and wants to know why.

**Step 1 - Global feature importance** (permutation importance on the held-out set). This shows which features drive predictions overall - probably `store_id`/`store_size`, lag features, `month`, then `promotion_type` and its interactions. This is useful background but doesn't yet explain those two specific recommendations.

**Step 2 - Local explanations using SHAP for the two predictions.** For each of the two store-months, I'd score all five candidate promotions and pull the SHAP values for the prediction the model made. SHAP breaks each prediction into per-feature contributions in the units of the target, so you can see exactly what's pushing the prediction up or down.

For **Store 12 in December**, the SHAP plot would probably show:
- A big positive contribution from `month = December` (festive baseline lift).
- A big positive contribution from `promotion_type = Loyalty Points Bonus` interacting with Store 12's demographics (likely an affluent, high-frequency clientele who already accumulate loyalty points). The model has learned that loyalty members in this store activate strongly when point multipliers are offered in the gift-buying season - they pre-load their accounts to spend on gifts.

For **Store 12 in March**, the SHAP plot would probably show:
- `Flat Discount` contributing strongly because March is a post-festive period where the same affluent customers are *less* responsive to perks and *more* responsive to clear monetary savings (they perceive flat discount as the simplest signal of value).

**How I'd communicate this to marketing.** Translate the SHAP output into business language: "in December, your loyalty members are gift-buying - they want to maximise the points they earn on gifts they were going to buy anyway. In March they're post-festive and value-shopping - they want to see the discount on the price tag." The model output becomes a *story* about customer behaviour, backed by the model's evidence, rather than a black-box command.

This is also a chance to validate the recommendation: ask the marketing team whether the historical March promotion mix for Store 12 has ever included a flat discount (probably not, which is why the recommendation feels surprising), and whether they'd be willing to do a small A/B test before a full rollout. Treating surprising recommendations as *hypotheses to test* rather than commands to obey is the right deployment posture.

### (c) Deployment, monitoring, retraining - 4 marks

The model needs to produce monthly recommendations for 50 stores on the 1st of each month, without a human retraining it every cycle.

**Deployment process end-to-end:**

1. **Batch scoring pipeline (runs on the 1st of every month).**
   - Pull the latest features for each store: store attributes (rarely change), `competition_density` (refresh from external feed), upcoming month's calendar features (deterministic), lag features (last month's actual `items_sold` from the data warehouse).
   - For each store, build five candidate feature rows - one per promotion option - and score all 250 rows through the saved model.
   - For each store, pick the promotion with the highest predicted items_sold (or apply business constraints: budget caps, head-office rules about which promotions can run together, store-level no-go lists).
   - Write the recommendations to a database table along with a confidence interval per recommendation (e.g. from quantile regression forest, or bootstrap intervals).
   - Surface the recommendations to marketing managers via a dashboard with the SHAP-based explanation alongside each one.

2. **Model artifact preparation.**
   - The trained pipeline (preprocessor + regressor) is saved once as a single object using `joblib.dump`. The same artifact is used for batch scoring and ad-hoc what-if queries.
   - Versioning is a must - every retraining produces a new versioned artifact, and the version that produced any given recommendation is logged for auditability.

3. **Feature pipeline reuse.** The exact same feature engineering code that built the training table runs in production - ideally invoked from the same module, not re-implemented. Skew between training and serving feature definitions is the most common cause of silent production failures.

**Monitoring - what I'd track to detect performance degradation:**

- **Realised vs predicted sales each month.** Once the actual `items_sold` for a month becomes available, compute RMSE and MAPE against the model's predictions for that month. Keep a rolling 6-month window and alert when it drifts outside the historical range from training.
- **Recommendation stability.** Track how often the model's recommended promotion changes for a given store from one month to the next. Sudden churn (a store that always got "BOGO" suddenly getting "Free Gift" with no obvious reason) is a sign of feature drift.
- **Feature drift on inputs.** Monitor the distribution of each input feature in scoring data vs training data, using PSI (Population Stability Index) or a simple KS test. New competitor openings, demographic shifts, pricing policy changes - all of these show up as feature drift before they show up as prediction error.
- **Top-1 accuracy on the realised promotion.** Where the marketing team did follow the recommendation, did the chosen promotion actually outperform the alternatives that *could* have been chosen? This is hard to evaluate (it's counterfactual), but periodic A/B tests (recommend the model's choice in 80% of stores, randomise across all five in the remaining 20%) keep this measurable.

**Retraining cadence:**

- **Scheduled retraining: every quarter.** Retrain on the most recent three years of data (rolling window). Three years balances having enough data to learn seasonal patterns vs staying current with consumer-preference shifts.
- **Triggered retraining:** automatically kick off a retrain if any of the rolling RMSE, PSI, or top-1 accuracy metrics breach their alert thresholds for two consecutive months. Don't retrain on a single bad month - single months can be anomalous (lockdowns, supply outages, weather events).
- **Pre-deployment validation gate.** Every retrained candidate model goes head-to-head against the previous champion on the most recent 3 months of held-out data. The new model only replaces the champion if it improves RMSE by a meaningful margin (say 2% or more) AND doesn't regress on any sub-segment (every `store_size` and `location_type` bucket has to hold or improve). Automatic rollback if a deployed model later underperforms its predecessor on the next month's actual data.

That combination - scheduled retraining, drift-triggered retraining, and a champion/challenger gate before any new model goes live - keeps the model fresh without exposing the business to silent failures or untested upgrades.
