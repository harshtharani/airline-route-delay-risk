# How this project works (plain language)

Read this after the project runs. Each section is something an interviewer might ask about.

## The question
Which airline routes are likely to be badly disrupted next month, and does knowing the network's shape help predict it?

## The data
- **BTS On-Time Performance:** one row per US flight with route, delay, cancellation and diversion.
- **OpenFlights:** airport coordinates, for maps and weather lookups.
- **Open-Meteo:** daily rain and wind at the busiest airports.

A flight counts as **disrupted** if it arrived 15+ minutes late, was cancelled or was diverted. Cancelled flights have no
delay number, so ignoring them would hide the worst cases. A route's **disruption rate** for a month is the share of its
flights that were disrupted.

## The graph
Each airport is a node and each route (origin to destination) is an edge, weighted by number of flights.
Network measures describe where an airport sits:
- **Degree:** how many routes it has.
- **PageRank:** importance, counting how important its neighbours are.
- **Betweenness:** how often it lies on shortest paths between other airports (a "bridge" role).
- **Clustering:** whether its neighbours are also connected to each other.
- **Top origin share:** how much of its inbound traffic comes from one source (dependency).
- **Articulation point / bridge route:** removing it would split the network.

## The GCN (graph convolutional network)
A GCN lets each airport "look at" its neighbours. Layer by layer it mixes an airport's numbers with its neighbours'
numbers, so the output is a short list of numbers (8 here) describing the airport's place in the network. We call this a
"structural fingerprint" or embedding.
It is **unsupervised**: it never sees delays. Its task is link prediction: given two airports, guess whether a route
connects them. To do that well it must capture who connects to whom. We train it on the first month's graph only, then reuse
it on later months, so nothing from the future leaks in. The AUC printed during training checks that it ranks real routes
above random airport pairs.

## The prediction problem
For each route, the features come from month t and the label is the risk class in month t+1.
- **Labels:** low / medium / high from terciles of the *training* label-month disruption rates. The same cut-offs are
  applied to the test month, so the test month never influences the definition of risk.
- **Split:** training uses early months; the test month comes last. This is a time-ordered split. A random split would let
  the model peek at the future, which is leakage.

## The models
1. **Persistence:** predict that next month's class equals this month's class. No learning. This is the bar to beat.
2. **XGBoost baseline:** gradient-boosted trees on route history, airport traffic and weather. No network features.
3. **XGBoost + graph:** the same plus centrality measures and the GCN embeddings.
Comparing 2 and 3 answers "does the graph help?". If it doesn't, that is a legitimate finding. Classes are balanced with
sample weights, and we report precision, recall and F1 per class because accuracy alone hides weak classes.

## Explainability (SHAP)
SHAP gives each feature a share of credit for each prediction. The dashboard shows the top three drivers per route, and
the importance chart shows which features matter overall.

## The hub-shutdown what-if
Remove one airport from the graph and measure: routes and share of flights lost, airports cut off from the main network,
longer detours (extra hops), and which airports depended most on it. It is structural only. It does not predict
how delays would actually spread.

## Confidence intervals (are the differences real?)
Two models can score 0.48 and 0.49 and still be equally good, because the test set is only a sample of routes.
The bootstrap re-draws the test routes at random thousands of times (with replacement), re-scores each model every time,
and reports the middle 95% of the results. If the interval for a *difference* between two models includes zero, the data
cannot tell them apart. This is how you can say "the graph features did not clearly help" with evidence.
Caveat: routes that share airports are not fully independent, so the intervals are a little too narrow.

## The forecast
`forecast.py` trains the same model on every month pair available, then scores the latest month to predict the month after.
Nothing can be checked against that month until it happens, so it is a prediction, not a result. The held-out test month
(a backtest) is the honest measure of accuracy. The dashboard says this on the Forecast tab.

## The dashboard tabs
- **Overview:** what is predicted, what low / medium / high mean, headline numbers and the key finding.
- **Model results:** the three approaches compared with confidence intervals, plus per-class scores and SHAP importance.
- **Route explorer:** predicted vs actual risk per route, with the main factors for each prediction.
- **Network map:** routes coloured by predicted risk, with the busiest airports labelled.
- **Forecast:** next month's predicted risk for every route.
- **Hub shutdown:** the structural what-if, plus a ranking of the 15 busiest airports by damage if closed.

## Neo4j
A graph database. It stores airports and routes, and (after training) each route's predicted risk, so you can query the
network in Cypher and draw it in Neo4j's visual tools.

## Things you should be able to say in an interview
- Why a time-ordered split, and what leakage is.
- Why the risk cut-offs come from training data only.
- What the GCN is trained to do and why it needs no delay labels.
- Why persistence is a fair baseline, and what your results say about the graph features.
- Why confidence intervals matter, and what your intervals say about the graph features.
- The difference between the backtest (checked) and the forecast (not yet checkable).
- The limitations: weather timing, few months, structural-only simulation, US data only.
