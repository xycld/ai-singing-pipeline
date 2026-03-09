pub struct Compressor {
    threshold_db: f64,
    ratio: f64,
    knee_db: f64,
    attack_coeff: f64,
    release_coeff: f64,
    env_db: f64,
}

impl Compressor {
    pub fn new(sr: f64, threshold_db: f64, ratio: f64, attack_ms: f64, release_ms: f64) -> Self {
        Self {
            threshold_db,
            ratio,
            knee_db: 6.0,
            attack_coeff: (-1.0 / (attack_ms * 0.001 * sr)).exp(),
            release_coeff: (-1.0 / (release_ms * 0.001 * sr)).exp(),
            env_db: -96.0,
        }
    }

    /// Apply feed-forward compression with soft-knee gain reduction.
    pub fn process_mono(&mut self, samples: &mut [f64]) {
        for s in samples.iter_mut() {
            let x_db = 20.0 * (s.abs() + 1e-8).log10();

            let gain_db = self.gain_computer(x_db);

            if gain_db < self.env_db {
                self.env_db =
                    self.attack_coeff * self.env_db + (1.0 - self.attack_coeff) * gain_db;
            } else {
                self.env_db =
                    self.release_coeff * self.env_db + (1.0 - self.release_coeff) * gain_db;
            }

            *s *= 10.0_f64.powf(self.env_db / 20.0);
        }
    }

    fn gain_computer(&self, x_db: f64) -> f64 {
        let t = self.threshold_db;
        let r = self.ratio;
        let w = self.knee_db;

        if x_db < t - w / 2.0 {
            0.0
        } else if x_db > t + w / 2.0 {
            (1.0 / r - 1.0) * (x_db - t)
        } else {
            let d = x_db - t + w / 2.0;
            (1.0 / r - 1.0) * d * d / (2.0 * w)
        }
    }
}
