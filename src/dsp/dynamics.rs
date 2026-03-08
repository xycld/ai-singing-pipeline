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
// Peak limiter (lookahead)
// ---------------------------------------------------------------------------
//
// Two-pass lookahead limiter modelled after broadcast/mastering limiters.
// Pass 1: scan the entire buffer and compute the minimum gain envelope
//         needed so that no sample exceeds `threshold`.
// Pass 2: apply that gain with smoothing (attack = lookahead, release = 100ms)
//         so gain changes are gradual and artifact-free.

pub struct Limiter {
    threshold: f64,
    lookahead: usize,
    release_coeff: f64,
}

impl Limiter {
    pub fn new(sr: f64, threshold_db: f64) -> Self {
        let lookahead_ms = 5.0;
        let release_ms = 100.0;
        Self {
            threshold: 10.0_f64.powf(threshold_db / 20.0),
            lookahead: (lookahead_ms * 0.001 * sr) as usize,
            release_coeff: (-1.0 / (release_ms * 0.001 * sr)).exp(),
        }
    }

    pub fn process_mono(&mut self, samples: &mut [f64]) {
        let n = samples.len();
        if n == 0 {
            return;
        }

        let threshold = self.threshold;
        let lookahead = self.lookahead.max(1);

        // Pass 1: compute per-sample target gain (<=1.0)
        let mut gain = vec![1.0_f64; n];
        for (i, s) in samples.iter().enumerate() {
            let abs = s.abs();
            if abs > threshold {
                gain[i] = threshold / abs;
            }
        }

        // Spread each gain reduction backward over the lookahead window
        // so the limiter ramps down *before* the peak arrives.
        // Walk backward: gain[i] must be <= gain[i+1] adjusted for ramp.
        let mut smoothed = gain.clone();
        for i in (0..n.saturating_sub(1)).rev() {
            let dist = (i + lookahead).min(n - 1);
            // Find minimum gain in the lookahead window ahead
            let mut min_g = smoothed[i];
            let mut j = i + 1;
            while j <= dist {
                if smoothed[j] < min_g {
                    min_g = smoothed[j];
                }
                j += 1;
            }
            if min_g < smoothed[i] {
                smoothed[i] = min_g;
            }
        }

        // Pass 2: smooth the gain envelope (slow release to avoid pumping)
        let mut current_gain = 1.0_f64;
        let release = self.release_coeff;
        for i in 0..n {
            let target = smoothed[i];
            if target < current_gain {
                // Attack: jump to target immediately (lookahead already handled ramp)
                current_gain = target;
            } else {
                // Release: exponential recovery
                current_gain = release * current_gain + (1.0 - release) * target;
            }
            samples[i] *= current_gain;
        }
    }
}
