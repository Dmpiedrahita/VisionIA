import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

datadir = r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2\Dataset500"

Categories=['Cat','Dog']
flat_data_arr=[] 
target_arr=[] 

for i in Categories:
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
      img_array=imread(os.path.join(path,img))
      img_resized=resize(img_array,(150,150,3))
      if img_resized.flatten().shape == (67500,):
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
        
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)

df=pd.DataFrame(flat_data)
df['Target']=target
print(df.shape)

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

model = svm.SVC(
    kernel="rbf",     
    C=10,             
    gamma=0.001,      
    probability=True,  
    random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)

print(f"The model is {accuracy*100}% accurate")
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

rand_index = np.random.randint(0, len(y_test))
rand_img = x_test.iloc[rand_index].values.reshape(150, 150, 3)
rand_class = y_test.iloc[rand_index]
pred_class = y_pred[rand_index]

plt.imshow(rand_img)
plt.title(f"Predicted: {Categories[pred_class]}, Actual: {Categories[rand_class]}")
plt.show()