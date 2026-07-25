from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state
        ),
        "SVM": SVC(
            probability=True,
            random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        ),
        "MLP": MLPClassifier(
            max_iter=1000,
            random_state=random_state
        )
    }