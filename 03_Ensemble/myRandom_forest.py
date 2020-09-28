from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=1, n_jobs=2, max_depth=10, min_samples_split=2, min_samples_leaf=1)
