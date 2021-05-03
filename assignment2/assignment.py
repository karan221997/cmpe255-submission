from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import seaborn as sns;

faces_of_personalities = fetch_lfw_people(min_faces_per_person=60)
print(faces_of_personalities.target_names)
print(faces_of_personalities.images.shape)

figure, axis = plt.subplots(3, 5)
for i, axi in enumerate(axis.flat):
    axi.imshow(faces_of_personalities.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces_of_personalities.target_names[faces_of_personalities.target[i]])

from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

principal_component_analysis = RandomizedPCA(n_components=150, whiten=True, random_state=42)
support_vector_classifier = SVC(kernel='rbf', class_weight='balanced')
m = make_pipeline(principal_component_analysis, support_vector_classifier)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces_of_personalities.data, faces_of_personalities.target,
                                                random_state=42)

from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(m, param_grid)

grid.fit(Xtrain, ytrain)
print(grid.best_params_)

m = grid.best_estimator_
yfit = m.predict(Xtest)

figure, axis = plt.subplots(4, 6)
for i, axi in enumerate(axis.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces_of_personalities.target_names[yfit[i]].split()[-1],
                   color='green' if yfit[i] == ytest[i] else 'red')
figure.suptitle('Predicted Names; Incorrect Labels(Red)', size=10)
plt.show()

from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces_of_personalities.target_names))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces_of_personalities.target_names,
            yticklabels=faces_of_personalities.target_names)
plt.xlabel('true ')
plt.ylabel('predicted ')
plt.show()