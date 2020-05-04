from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense
from tensorflow.keras import Model, Input
from utils import create_dataset
from tensorflow.keras.utils import to_categorical

X_train, y_train, X_test, y_test = create_dataset()
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


input = Input(shape=(40, 40, 1))

x = Conv2D(8, (3, 3), activation='relu')(input)
x = MaxPool2D(pool_size=(3, 3), strides=(3, 3))(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
output = Dense(2, activation='sigmoid')(x)

model = Model(input, output)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=1, verbose=2,
          validation_data=(X_test, y_test))
