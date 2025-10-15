import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# تحميل بيانات CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# تطبيع الصور (من 0-255 إلى 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# أسماء الفئات
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# بناء نموذج CNN بسيط
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# تجميع النموذج
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# تدريب النموذج
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

# حفظ النموذج
model.save('image_classifier_model.h5')

# تقييم الأداء
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n✅ دقة النموذج على بيانات الاختبار: {test_acc:.2f}')
