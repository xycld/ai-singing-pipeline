//! Noise gate (downward expander).
//!
//! Attenuates signal below threshold by the given ratio.

pub struct Gate {
    threshold_db: f64,
    ratio: f64,
    attack_coeff: f64,
    release_coeff: f64,
    env_db: f64,
}

impl Gate {
    pub fn new(sr: f64, threshold_db: f64, ratio: f64, release_ms: f64) -> Self {
        let attack_ms = 0.1; // near-instant attack for gates
        Self {
            threshold_db,
            ratio,
            attack_coeff: (-1.0 / (attack_ms * 0.001 * sr)).exp(),
            release_coeff: (-1.0 / (release_ms * 0.001 * sr)).exp(),
            env_db: 0.0,
        }
    }

    pub fn process_mono(&mut self, samples: &mut [f64]) {
        for s in samples.iter_mut() {
            let x_db = 20.0 * (s.abs() + 1e-8).log10();

            // Below threshold: apply expansion (reduce gain)
            let gain_db = if x_db < self.threshold_db {
                (1.0 - self.ratio) * (x_db - self.threshold_db)
            } else {
                0.0
            };

            // Ballistic smoothing
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
}
