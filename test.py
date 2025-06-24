# 作者:王勇
# 开发时间:2024/3/21 15:28
import tensorflow as tf
from keras.layers import Input, Conv2D, GlobalAveragePooling2D
from keras.models import Model

# 假设你有一个形状为[batch_size, features]的二维张量
batch_size = 32
features = 64
x = tf.random.normal([batch_size, features])  # 示例数据

# 将二维数据扩展为四维数据，增加高度和宽度的维度，并设置为1
x_expanded = tf.expand_dims(x, axis=-2)  # 在倒数第二个维度上增加维度，形成[batch_size, features, 1]
x_expanded = tf.expand_dims(x_expanded, axis=-2)  # 在倒数第二个维度上再次增加维度，形成[batch_size, features, 1, 1]

# 注意：如果你的数据原本就有通道维度（比如彩色图像的RGB通道），则只需要增加空间维度即可。
# 但在这个例子中，我们的features对应的是通道数，所以不需要再次扩展通道维度。

# 现在你可以将x_expanded用作1x1卷积层的输入
input_layer = Input(shape=(features, 1, 1))
conv_layer = Conv2D(filters=10, kernel_size=(1, 1))(input_layer)
model = Model(inputs=input_layer, outputs=conv_layer)

# 使用模型对扩展后的数据进行预测（这里只是为了演示，实际上你应该在训练循环中使用这些数据）
output = model.predict(x_expanded)
print("Output shape:", output.shape)  # 应该会输出类似于[batch_size, 10, 1, 1]的形状

# 如果你希望去掉额外的维度以匹配全连接层的输出，可以使用squeeze操作：
output_squeezed = tf.squeeze(output, axis=[1, 2])  # 去掉高度和宽度的维度
print("Squeezed output shape:", output_squeezed.shape)  # 应该会输出[batch_size, 10]的形状