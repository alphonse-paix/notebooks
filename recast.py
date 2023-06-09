import numpy as np
import tensorflow as tf
from tensorflow import keras


class Weibull:
    def __init__(self, b, k, eps=1e-8):
        self.b = tf.constant(b, dtype=tf.float32)
        self.k = tf.constant(k, dtype=tf.float32)
        self.eps = eps


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
    

class Gamma:
    def __init__(self, alpha, beta, eps=1e-8):
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.eps = eps


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
    def __init__(self, context_size=32, dist=Weibull):
        self.context_size = context_size
        self.encoder = keras.layers.GRU(context_size, return_sequences=True)
        self.decoder = keras.layers.Dense(2, activation="softplus")
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.dist = dist
        self.dist_params = {"b": [], "k": []}


    def get_context(self, inter_times):
        tau = tf.expand_dims(inter_times, axis=-1)
        log_tau = tf.math.log(tf.clip_by_value(tau, 1e-8, tf.reduce_max(tau)))
        input = tf.concat([tau, log_tau], axis=-1)
        output = self.encoder(input)
        context = tf.pad(output[:, :-1, :], [[0, 0], [1, 0], [0, 0]])
        return context


    def get_inter_times_distribution(self, context):
        if self.dist == Weibull or self.dist == Gamma:
            params = self.decoder(context)
            b = params[..., 0]
            k = params[..., 1]
            self.dist_params["b"].append(b)
            self.dist_params["k"].append(k)
            return self.dist(b, k)
        else:
            assert False and "Distribution not supported"


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
        inter_time = inter_times[-1]
        preds = []
        for _ in range(num_preds):
            last = inter_time
            tau = tf.expand_dims(last, axis=-1)
            log_tau = tf.math.log(
                tf.clip_by_value(tau, 1e-8, tf.reduce_max(tau)))
            input = tf.concat([tau, log_tau], axis=-1)
            context = self.encoder(input)
            dist = self.get_inter_times_distribution(context)
            inter_time = dist.sample(1)
            preds.append(inter_time)
        return inter_times[-1] + np.cumsum(preds)
