from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis
iris = load_iris()
fa = FactorAnalysis(n_components=2)
iris_two_dim = fa.fit_transform(iris.data)
iris_two_dim[:5]
print(iris_two_dim)