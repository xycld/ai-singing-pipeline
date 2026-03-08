//! Reverb (Freeverb algorithm) and peak limiter.

// ---------------------------------------------------------------------------
// Freeverb
// ---------------------------------------------------------------------------

const COMB_DELAYS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASS_DELAYS: [usize; 4] = [556, 441, 341, 225];
const FIXED_GAIN: f64 = 0.015;

struct CombFilter {
    buf: Vec<f64>,
    idx: usize,
    feedback: f64,
    damp1: f64,
    damp2: f64,
    filterstore: f64,
}

impl CombFilter {
    fn new(delay: usize) -> Self {
        Self {
            buf: vec![0.0; delay],
            idx: 0,
            feedback: 0.5,
            damp1: 0.5,
            damp2: 0.5,
            filterstore: 0.0,
        }
    }

    fn set_feedback(&mut self, val: f64) {
        self.feedback = val;
    }

    fn set_damp(&mut self, val: f64) {
        self.damp1 = val;
        self.damp2 = 1.0 - val;
    }

    fn process(&mut self, input: f64) -> f64 {
        let output = self.buf[self.idx];
        self.filterstore = output * self.damp2 + self.filterstore * self.damp1;
        self.buf[self.idx] = input + self.filterstore * self.feedback;
        self.idx = (self.idx + 1) % self.buf.len();
        output
    }
}

struct AllpassFilter {
    buf: Vec<f64>,
    idx: usize,
}

impl AllpassFilter {
    fn new(delay: usize) -> Self {
        Self {
            buf: vec![0.0; delay],
            idx: 0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let bufout = self.buf[self.idx];
        let output = -input + bufout;
        self.buf[self.idx] = input + bufout * 0.5;
        self.idx = (self.idx + 1) % self.buf.len();
        output
    }
}

pub struct Reverb {
    combs: Vec<CombFilter>,
    allpasses: Vec<AllpassFilter>,
    wet: f64,
    dry: f64,
}

impl Reverb {
    /// Create a Freeverb instance.
    ///
    /// - `room_size` 0.0–1.0 (scales feedback)
    /// - `wet` / `dry` mix levels
    pub fn new(sr: f64, room_size: f64, wet: f64, dry: f64) -> Self {
        // Scale delays for sample rates other than 44100
        let scale = sr / 44100.0;
        let damping = 0.5;
        let feedback = 0.28 + room_size * 0.7; // map 0..1 → 0.28..0.98

        let mut combs: Vec<CombFilter> = COMB_DELAYS
            .iter()
            .map(|&d| {
                let delay = ((d as f64) * scale) as usize;
                let mut c = CombFilter::new(delay.max(1));
                c.set_feedback(feedback);
                c.set_damp(damping);
                c
            })
            .collect();

        // Slightly detune odd combs for stereo-ish width even in mono
        for (i, c) in combs.iter_mut().enumerate() {
            if i % 2 == 1 {
                c.set_feedback(feedback * 0.98);
            }
        }

        let allpasses: Vec<AllpassFilter> = ALLPASS_DELAYS
            .iter()
            .map(|&d| {
                let delay = ((d as f64) * scale) as usize;
                AllpassFilter::new(delay.max(1))
            })
            .collect();

        Self {
            combs,
            allpasses,
            wet,
            dry,
        }
    }

    pub fn process_mono(&mut self, samples: &mut [f64]) {
        for s in samples.iter_mut() {
            let input = *s * FIXED_GAIN;

            // Parallel comb filters
            let mut comb_sum = 0.0;
            for comb in self.combs.iter_mut() {
                comb_sum += comb.process(input);
            }

            // Series allpass filters
            let mut out = comb_sum;
            for ap in self.allpasses.iter_mut() {
                out = ap.process(out);
            }

            *s = self.dry * *s + self.wet * out;
        }
    }
}

// ---------------------------------------------------------------------------
// Peak limiter
// ---------------------------------------------------------------------------

pub struct Limiter {
    threshold: f64,
    attack_coeff: f64,
    release_coeff: f64,
    env: f64,
}

impl Limiter {
    pub fn new(sr: f64, threshold_db: f64) -> Self {
        let attack_ms = 0.1;
        let release_ms = 50.0;
        Self {
            threshold: 10.0_f64.powf(threshold_db / 20.0),
            attack_coeff: (-1.0 / (attack_ms * 0.001 * sr)).exp(),
            release_coeff: (-1.0 / (release_ms * 0.001 * sr)).exp(),
            env: 0.0,
        }
    }

    pub fn process_mono(&mut self, samples: &mut [f64]) {
        for s in samples.iter_mut() {
            let abs = s.abs();
            // Envelope follower: fast attack, slow release
            if abs > self.env {
                self.env = self.attack_coeff * self.env + (1.0 - self.attack_coeff) * abs;
            } else {
                self.env = self.release_coeff * self.env + (1.0 - self.release_coeff) * abs;
            }

            // Apply gain reduction when envelope exceeds threshold
            if self.env > self.threshold {
                *s *= self.threshold / self.env;
            }
        }
    }
}
