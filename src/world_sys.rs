//! Raw FFI bindings for WORLD vocoder C library.

#[repr(C)]
pub struct HarvestOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub frame_period: f64,
}

#[repr(C)]
pub struct CheapTrickOption {
    pub q1: f64,
    pub f0_floor: f64,
    pub fft_size: i32,
}

#[repr(C)]
pub struct D4COption {
    pub threshold: f64,
}

extern "C" {
    // Harvest — F0 estimation
    pub fn InitializeHarvestOption(option: *mut HarvestOption);
    pub fn GetSamplesForHarvest(fs: i32, x_length: i32, frame_period: f64) -> i32;
    pub fn Harvest(
        x: *const f64,
        x_length: i32,
        fs: i32,
        option: *const HarvestOption,
        temporal_positions: *mut f64,
        f0: *mut f64,
    );

    // CheapTrick — spectral envelope
    pub fn InitializeCheapTrickOption(fs: i32, option: *mut CheapTrickOption);
    pub fn GetFFTSizeForCheapTrick(fs: i32, option: *const CheapTrickOption) -> i32;
    pub fn CheapTrick(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        option: *const CheapTrickOption,
        spectrogram: *mut *mut f64,
    );

    // D4C — aperiodicity
    pub fn InitializeD4COption(option: *mut D4COption);
    pub fn D4C(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        fft_size: i32,
        option: *const D4COption,
        aperiodicity: *mut *mut f64,
    );

    // Synthesis
    pub fn Synthesis(
        f0: *const f64,
        f0_length: i32,
        spectrogram: *const *const f64,
        aperiodicity: *const *const f64,
        fft_size: i32,
        frame_period: f64,
        fs: i32,
        y_length: i32,
        y: *mut f64,
    );
}
