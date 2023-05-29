import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle

class Distribution:
    def __init__(self, name, *params):
        self.name = name
        self.d_params = params
        self.eps = 1e-8


    @property
    def params(self):
        return self.d_params
    

class Weibull(Distribution):
    def __init__(self, *params):
        if params:
            assert(len(params) == 2 and "Wrong number of parameters")
            self.b = params[0]
            self.k = params[1]
        super().__init__("weibull", *params)
        self.n_params = 2


    def __call__(self, *params):
        return Weibull(*params)


    def prob(self, x):
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        return self.b * self.k * tf.math.pow(x, self.k - 1) \
            * tf.math.exp(-self.b * tf.math.pow(x, self.k))


    def log_prob(self, x):
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        return tf.math.log(self.b) + tf.math.log(self.k) \
            + (self.k - 1) * tf.math.log(x) - self.b * tf.math.pow(x, self.k)
    

    def log_survival(self, x):
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        return -self.b * tf.math.pow(x, self.k)
    

    def sample(self, size=()):            
        u = tf.random.uniform(minval=0,
                              maxval=1,
                              shape=self.b.shape + size,
                              dtype=tf.float32)
        y = -1 / self.b * tf.math.log(1 - u)
        y = tf.clip_by_value(y, self.eps, tf.reduce_max(y))
        return tf.math.pow(y, 1 / self.k).numpy()
    

class Gamma(Distribution):
    def __init__(self, *params):
        super().__init__("gaussian", *params)


    def prob(self, x):
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        return tf.math.pow(x, self.alpha - 1) \
            * tf.math.pow(self.beta, self.alpha) \
            * tf.math.exp(-self.beta * x) \
            / tf.math.exp(tf.math.lgamma(self.alpha))


    def log_prob(self, x):
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        self.alpha = tf.clip_by_value(self.alpha, self.eps,
                                      tf.reduce_max(self.alpha))
        self.beta = tf.clip_by_value(self.beta, self.eps,
                                     tf.reduce_max(self.beta))
        return (self.alpha - 1) * tf.math.log(x) \
            + self.alpha * tf.math.log(self.beta) - self.beta * x \
            - tf.math.lgamma(self.alpha)

    
    def survival(self, x):
        rhs = self.beta * x
        rhs = tf.clip_by_value(rhs, self.eps, tf.reduce_max(rhs))
        self.alpha = tf.clip_by_value(self.alpha, self.eps,
                                      tf.reduce_max(self.alpha))
        self.beta = tf.clip_by_value(self.beta, self.eps,
                                     tf.reduce_max(self.beta))
        return 1 - tf.math.igamma(self.alpha, rhs) \
            / tf.math.exp(tf.math.lgamma(self.alpha))


    def log_survival(self, x):
        y = self.survival(x)
        y = tf.clip_by_value(y, self.eps, tf.reduce_max(y))
        return tf.math.log(y)


    def sample(self, size=()):            
        return tf.random.gamma(size, self.alpha, self.beta).numpy()


class Model:
    def __init__(self, context_size=32, dist=Distribution):
        self.context_size = context_size
        self.encoder = keras.layers.GRU(context_size, return_sequences=True)
        self.decoder = keras.layers.Dense(dist.n_params, activation="softplus")
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.dist = dist


    def get_context(self, inter_times):
        tau = tf.expand_dims(inter_times, axis=-1)
        log_tau = tf.math.log(tf.clip_by_value(tau, 1e-8, tf.reduce_max(tau)))
        input = tf.concat([tau, log_tau], axis=-1)
        output = self.encoder(input)
        context = tf.pad(output[:, :-1, :], [[0, 0], [1, 0], [0, 0]])
        return context


    def get_inter_times_distribution(self, context):
        params = self.decoder(context)
        b = params[..., 0]
        k = params[..., 1]
        return self.dist(b, k)


    def nll_loss(self, inter_times, seq_lengths):
        context = self.get_context(inter_times)
        inter_times_dist = self.get_inter_times_distribution(context)

        log_pdf = inter_times_dist.log_prob(inter_times)
        log_surv = inter_times_dist.log_survival(inter_times)

        # construit un masque pour ne sélectionner que les éléments
        # nécessaires dans chaque liste
        mask = np.cumsum(np.ones_like(log_pdf), axis=-1) \
            <= np.expand_dims(seq_lengths, axis=-1)
        log_like = tf.reduce_sum(log_pdf * mask, axis=-1)
        
        # idx est une liste de la forme [(a1, b1), (a2, b2), ...]
        # gather_nd sélectionne les éléments correspondant à ces indices
        # (ligne et colonne)
        idx = list(zip(range(len(seq_lengths)), seq_lengths))
        log_surv_last = tf.gather_nd(log_surv, idx)
        log_like += log_surv_last

        return -log_like
    

    @property
    def weights(self):
        return self.encoder.trainable_weights + self.decoder.trainable_weights
        
    
    def fit(self, epochs, inter_times, seq_lengths, t_end):
        for epoch in range(epochs + 1):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.nll_loss(inter_times,
                                                    seq_lengths)) / t_end
            grads = tape.gradient(loss, self.weights)
            self.optimizer.apply_gradients(zip(grads, self.weights))

            if epoch % 10 == 0:
                print(f"Loss at epoch {epoch}: {loss:.2f}")


    def sample(self, batch_size, t_end):
        inter_times = np.empty((batch_size, 0))
        next_context = tf.zeros(shape=(batch_size, 1, 32))
        generated = False

        while not generated:
            dist = self.get_inter_times_distribution(next_context)
            next_inter_times = dist.sample()
            inter_times = tf.concat([inter_times, next_inter_times], axis=-1)
            tau = tf.expand_dims(next_inter_times, axis=-1)
            log_tau = tf.math.log(
                tf.clip_by_value(tau, 1e-8, tf.reduce_max(tau)))
            input = tf.concat([tau, log_tau], axis=-1)
            next_context = self.encoder(input)

            generated = np.sum(inter_times, axis=-1).min() >= t_end

        return np.cumsum(inter_times, axis=-1)
    

    def next(self, inter_times, num_preds=1):
        for _ in range(num_preds):
            next_context = self.get_context(inter_times)


file = open("data/shchur.pkl", "rb")
data = pickle.load(file)
t_end = data["t_end"]
arrival_times = data["arrival_times"]
seq_lengths = [len(times) for times in arrival_times]
inter_times_list = [np.diff(times, prepend=0, append=t_end)
                    for times in arrival_times]
inter_times = np.asarray([np.pad(inter_times, (0, np.max(seq_lengths) - size))
        for size, inter_times in zip(seq_lengths, inter_times_list)])
inter_times = tf.Variable(inter_times, dtype=tf.float32)

model = Model(32, Weibull())
model.fit(120, inter_times, seq_lengths, t_end)