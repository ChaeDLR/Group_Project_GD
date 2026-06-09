from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=0)
x = [[1, 2, 3], [11, 12, 13]]
y = [0, 1]

if __name__ == "__main__":
    clf.fit(x, y)
    p_x = clf.predict(x)  # predict classes of the training data
    p_new = clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data

    print(f"Training data predictions: {p_x}")
    print(f"New data predictions: {p_new}")
