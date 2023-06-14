from tensorflow.keras.layers import Input, Conv2DTranspose,concatenate, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, Dense, Multiply, Add, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model


class Resunetpp:
    def __init__(self, sz=(256, 256, 3)):
        self.sz = sz
        self.n_filters = [16, 32, 64, 128, 256]
        self.model = self.build_model()

    def squeeze_excite_block(self, inputs, ratio=8):
        init = inputs
        channel_axis = -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        x = Multiply()([init, se])
        return x

    def stem_block(self, x, n_filter, strides):
        x_init = x

        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same")(x)

        s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        x = Add()([x, s])
        x = self.squeeze_excite_block(x)
        return x

    def resnet_block(self, x, n_filter, strides=1):
        x_init = x

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

        s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        x = Add()([x, s])
        x = self.squeeze_excite_block(x)
        return x

    def aspp_block(self, x, num_filters, rate_scale=1):
        x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
        x2 = BatchNormalization()(x2)

        x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
        x3 = BatchNormalization()(x3)

        x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
        x4 = BatchNormalization()(x4)

        y = Add()([x1, x2, x3, x4])
        y = Conv2D(num_filters, (1, 1), padding="same")(y)
        return y

    def attention_block(self, g, x):
        filters = x.shape[-1]

        g_conv = BatchNormalization()(g)
        g_conv = Activation("relu")(g_conv)
        g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

        g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

        x_conv = BatchNormalization()(x)
        x_conv = Activation("relu")(x_conv)
        x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

        gc_sum = Add()([g_pool, x_conv])

        gc_conv = BatchNormalization()(gc_sum)
        gc_conv = Activation("relu")(gc_conv)
        gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

        gc_mul = Multiply()([gc_conv, x])
        return gc_mul

    def build_model(self):
        x = Input(self.sz)
        c0 = x
        c1 = self.stem_block(c0, self.n_filters[0], strides=1)

        c2 = self.resnet_block(c1, self.n_filters[1], strides=2)
        c3 = self.resnet_block(c2, self.n_filters[2], strides=2)
        c4 = self.resnet_block(c3, self.n_filters[3], strides=2)

        b1 = self.aspp_block(c4, self.n_filters[4])

        d1 = self.attention_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = self.resnet_block(d1, self.n_filters[3])

        d2 = self.attention_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = self.resnet_block(d2, self.n_filters[2])

        d3 = self.attention_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = self.resnet_block(d3, self.n_filters[1])

        outputs = self.aspp_block(d3, self.n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        model = Model(x, outputs)
        return model


class Resunet:
    def __init__(self, img_h=256, img_w=256):
        self.img_h = img_h
        self.img_w = img_w
        self.f = [16, 32, 64, 128, 256]
        self.model = self.build_model()

    def bn_act(self, x, act=True):
        x = BatchNormalization()(x)
        if act:
            x = Activation('relu')(x)
        return x

    def conv_block(self, x, filters, kernel_size=3, padding='same', strides=1):
        conv = self.bn_act(x)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(self, x, filters, kernel_size=3, padding='same', strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size, padding, strides)
        shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        output = Add()([conv, shortcut])
        return output

    def residual_block(self, x, filters, kernel_size=3, padding='same', strides=1):
        res = self.conv_block(x, filters, kernel_size, padding, strides)
        res = self.conv_block(res, filters, kernel_size, padding, 1)
        shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        output = Add()([shortcut, res])
        return output

    def upsample_concat_block(self, x, xskip):
        u = UpSampling2D((2, 2))(x)
        c = Concatenate()([u, xskip])
        return c

    def build_model(self):
        inputs = Input((self.img_h, self.img_w, 3))

        e0 = inputs
        e1 = self.stem(e0, self.f[0])
        e2 = self.residual_block(e1, self.f[1], strides=2)
        e3 = self.residual_block(e2, self.f[2], strides=2)
        e4 = self.residual_block(e3, self.f[3], strides=2)
        e5 = self.residual_block(e4, self.f[4], strides=2)

        b0 = self.conv_block(e5, self.f[4], strides=1)
        b1 = self.conv_block(b0, self.f[4], strides=1)

        u1 = self.upsample_concat_block(b1, e4)
        d1 = self.residual_block(u1, self.f[4])

        u2 = self.upsample_concat_block(d1, e3)
        d2 = self.residual_block(u2, self.f[3])

        u3 = self.upsample_concat_block(d2, e2)
        d3 = self.residual_block(u3, self.f[2])

        u4 = self.upsample_concat_block(d3, e1)
        d4 = self.residual_block(u4, self.f[1])

        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
        model = Model(inputs, outputs)
        return model
    

class Unet:
    def __init__(self, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        self.model = self.build_model()

    def build_model(self):
        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        s = inputs

        # Contraction path
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path bin
        u6_bin = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6_bin = concatenate([u6_bin, c4])
        c6_bin = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6_bin)
        c6_bin = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6_bin)

        u7_bin = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6_bin)
        u7_bin = concatenate([u7_bin, c3])
        c7_bin = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7_bin)
        c7_bin = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7_bin)

        u8_bin = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7_bin)
        u8_bin = concatenate([u8_bin, c2])
        c8_bin = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8_bin)
        c8_bin = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8_bin)

        u9_bin = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8_bin)
        u9_bin = concatenate([u9_bin, c1], axis=3)
        c9_bin = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9_bin)
        c9_bin = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9_bin)

        bin_seg = Conv2D(1, (1, 1), activation='sigmoid', name='bin_seg')(c9_bin)

        model = Model(inputs=[inputs], outputs=[bin_seg])

        return model