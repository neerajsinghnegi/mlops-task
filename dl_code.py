from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
def add_layer(model,neuron)
	model.add(Dense(neuron,activation = 'relu'))

dataset = loadtxt(r'/usr/pima-indians-diabetes.csv', delimiter= ',')
neuron=8
epoch=20
X = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
add_layer(model,neuron)
model.summary()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = binary_crossentropy, optimizer='adam', metrics=['accuracy'])
model.fit(X,y, epoch = epoch, batch_size = 10))
_, accuracy = model.evaluate(X,y)
print('Accuracy: %.2f % (accuracy*100)')
os.system('echo {} | cat > /usr/accuracy.txt'.format(str((accuracy*100))))