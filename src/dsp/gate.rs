//! Noise gate (downward expander).
//!
//! Attenuates signal below threshold by the given ratio.
//!
//! Uses standard gate terminology:
//! - **attack** = how fast the gate *opens* (signal returns above threshold)
//! - **release** = how fast the gate *closes* (signal drops below threshold)

pub struct Gate {
    threshold_db: f64,
    ratio: f64,
    /// Smoothing coeff for gate opening (signal back above threshold).
    open_coeff: f64,
    /// Smoothing coeff for gate closing (signal drops below threshold).
    close_coeff: f64,
    env_db: f64,
}

impl Gate {
    pub fn new(sr: f64, threshold_db: f64, ratio: f64, release_ms: f64) -> Self {
        // Pedalboard default: attack 1ms, release as given.
        // attack = gate opens fast, release = gate closes slowly.
        let attack_ms = 1.0;
        Self {
            threshold_db,
            ratio,
            open_coeff: (-1.0 / (attack_ms * 0.001 * sr)).exp(),
            close_coeff: (-1.0 / (release_ms * 0.001 * sr)).exp(),
            env_db: 0.0,
        }
    }

    pub fn process_mono(&mut self, samples: &mut [f64]) {
        for s in samples.iter_mut() {
            let x_db = 20.0 * (s.abs() + 1e-8).log10();

            // Below threshold: expand downward (attenuate further)
            // y = threshold + ratio*(x - threshold), so gain = (ratio-1)*(x-threshold) < 0
            let gain_db = if x_db < self.threshold_db {
                (self.ratio - 1.0) * (x_db - self.threshold_db)
            } else {
                0.0
            };

            // Ballistic smoothing — gate semantics:
            //   gain_db < env_db → more attenuation → gate closing → slow (release)
            //   gain_db > env_db → less attenuation → gate opening → fast (attack)
            if gain_db < self.env_db {
                self.env_db =
                    self.close_coeff * self.env_db + (1.0 - self.close_coeff) * gain_db;
            } else {
                self.env_db =
                    self.open_coeff * self.env_db + (1.0 - self.open_coeff) * gain_db;
            }

            *s *= 10.0_f64.powf(self.env_db / 20.0);
        }
    }
}
