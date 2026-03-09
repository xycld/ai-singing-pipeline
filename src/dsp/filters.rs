use std::f64::consts::PI;

/// Biquad IIR filter (Direct Form II Transposed).
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    /// Second-order Butterworth high-pass filter.
    pub fn highpass(sr: f64, cutoff: f64) -> Self {
        let w0 = 2.0 * PI * cutoff / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // Q = 1/sqrt(2) for Butterworth
        let alpha = sin_w0 / (2.0 * 2.0_f64.sqrt());

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    /// High-shelf filter. Positive `gain_db` boosts, negative cuts.
    pub fn high_shelf(sr: f64, freq: f64, gain_db: f64) -> Self {
        let a_lin = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * 2.0_f64.sqrt();
        let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;

        let b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
        let b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
        let a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    /// Parametric peak EQ filter.
    pub fn peak_eq(sr: f64, freq: f64, gain_db: f64, q: f64) -> Self {
        let a_lin = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0 + alpha * a_lin;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a_lin;
        let a0 = 1.0 + alpha / a_lin;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a_lin;

        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    fn from_coeffs(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Filter samples in-place.
    pub fn process_mono(&mut self, samples: &mut [f64]) {
        for s in samples.iter_mut() {
            let x = *s;
            let y = self.b0 * x + self.z1;
            self.z1 = self.b1 * x - self.a1 * y + self.z2;
            self.z2 = self.b2 * x - self.a2 * y;
            *s = y;
        }
    }

    /// Reset filter state to zero.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}
