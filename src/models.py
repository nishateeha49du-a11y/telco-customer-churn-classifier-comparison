from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier


def get_models(random_state=42):

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=random_state
        ),

        "LDA": LinearDiscriminantAnalysis(),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Gaussian Naive Bayes": GaussianNB(),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),

        "SVM": CalibratedClassifierCV(
            estimator=SVC(
                random_state=random_state
            ),
            ensemble=False
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

    return models