//! Safe Rust wrappers around WORLD vocoder FFI.

use crate::world_sys;
use std::alloc::{self, Layout};

/// F0 estimation via Harvest algorithm.
/// Returns (temporal_positions, f0).
pub fn harvest(audio: &[f64], fs: i32, frame_period: f64) -> (Vec<f64>, Vec<f64>) {
    let x_length = audio.len() as i32;
    let n_frames =
        unsafe { world_sys::GetSamplesForHarvest(fs, x_length, frame_period) } as usize;

    let mut option = unsafe { std::mem::zeroed::<world_sys::HarvestOption>() };
    unsafe { world_sys::InitializeHarvestOption(&mut option) };
    option.frame_period = frame_period;

    let mut times = vec![0.0f64; n_frames];
    let mut f0 = vec![0.0f64; n_frames];

    unsafe {
        world_sys::Harvest(
            audio.as_ptr(),
            x_length,
            fs,
            &option,
            times.as_mut_ptr(),
            f0.as_mut_ptr(),
        );
    }

    (times, f0)
}

/// Spectral envelope estimation via CheapTrick.
/// Returns (spectrogram as Vec<Vec<f64>>, fft_size).
pub fn cheaptrick(
    audio: &[f64],
    fs: i32,
    times: &[f64],
    f0: &[f64],
) -> (Vec<Vec<f64>>, i32) {
    let f0_length = f0.len() as i32;
    let x_length = audio.len() as i32;

    let mut option = unsafe { std::mem::zeroed::<world_sys::CheapTrickOption>() };
    unsafe { world_sys::InitializeCheapTrickOption(fs, &mut option) };
    let fft_size = unsafe { world_sys::GetFFTSizeForCheapTrick(fs, &option) };
    option.fft_size = fft_size;

    let spec_len = (fft_size / 2 + 1) as usize;
    let n_frames = f0.len();

    // Allocate 2D array: array of row pointers, each row is spec_len doubles
    let mut row_ptrs = alloc_2d(n_frames, spec_len);

    unsafe {
        world_sys::CheapTrick(
            audio.as_ptr(),
            x_length,
            fs,
            times.as_ptr(),
            f0.as_ptr(),
            f0_length,
            &option,
            row_ptrs.as_mut_ptr(),
        );
    }

    let spectrogram = copy_2d_to_vecs(&row_ptrs, spec_len);
    free_2d(row_ptrs, spec_len);

    (spectrogram, fft_size)
}

/// Aperiodicity estimation via D4C.
pub fn d4c(
    audio: &[f64],
    fs: i32,
    times: &[f64],
    f0: &[f64],
    fft_size: i32,
) -> Vec<Vec<f64>> {
    let f0_length = f0.len() as i32;
    let x_length = audio.len() as i32;
    let spec_len = (fft_size / 2 + 1) as usize;
    let n_frames = f0.len();

    let mut option = unsafe { std::mem::zeroed::<world_sys::D4COption>() };
    unsafe { world_sys::InitializeD4COption(&mut option) };

    let mut row_ptrs = alloc_2d(n_frames, spec_len);

    unsafe {
        world_sys::D4C(
            audio.as_ptr(),
            x_length,
            fs,
            times.as_ptr(),
            f0.as_ptr(),
            f0_length,
            fft_size,
            &option,
            row_ptrs.as_mut_ptr(),
        );
    }

    let aperiodicity = copy_2d_to_vecs(&row_ptrs, spec_len);
    free_2d(row_ptrs, spec_len);

    aperiodicity
}

/// WORLD synthesis from f0 + spectral envelope + aperiodicity.
pub fn synthesis(
    f0: &[f64],
    spectrogram: &[Vec<f64>],
    aperiodicity: &[Vec<f64>],
    fft_size: i32,
    frame_period: f64,
    fs: i32,
) -> Vec<f64> {
    let f0_length = f0.len() as i32;
    // Output length: same formula WORLD uses internally
    let y_length = ((f0.len() as f64 - 1.0) * frame_period * fs as f64 / 1000.0) as i32 + 1;

    let spec_ptrs: Vec<*const f64> = spectrogram.iter().map(|row| row.as_ptr()).collect();
    let aper_ptrs: Vec<*const f64> = aperiodicity.iter().map(|row| row.as_ptr()).collect();

    let mut y = vec![0.0f64; y_length as usize];

    unsafe {
        world_sys::Synthesis(
            f0.as_ptr(),
            f0_length,
            spec_ptrs.as_ptr(),
            aper_ptrs.as_ptr(),
            fft_size,
            frame_period,
            fs,
            y_length,
            y.as_mut_ptr(),
        );
    }

    y
}

// --- 2D allocation helpers for C interop ---

fn alloc_2d(rows: usize, cols: usize) -> Vec<*mut f64> {
    let mut ptrs = Vec::with_capacity(rows);
    let layout = Layout::array::<f64>(cols).unwrap();
    for _ in 0..rows {
        let ptr = unsafe { alloc::alloc_zeroed(layout) as *mut f64 };
        assert!(!ptr.is_null(), "allocation failed");
        ptrs.push(ptr);
    }
    ptrs
}

fn copy_2d_to_vecs(ptrs: &[*mut f64], cols: usize) -> Vec<Vec<f64>> {
    ptrs.iter()
        .map(|&ptr| unsafe { std::slice::from_raw_parts(ptr, cols).to_vec() })
        .collect()
}

fn free_2d(ptrs: Vec<*mut f64>, cols: usize) {
    let layout = Layout::array::<f64>(cols).unwrap();
    for ptr in ptrs {
        unsafe { alloc::dealloc(ptr as *mut u8, layout) };
    }
}
